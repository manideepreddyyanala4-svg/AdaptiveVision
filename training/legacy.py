"""The original single-model training path: autoencoder and PaDiM, one category at a time.

This predates ``training.sweep`` (the multi-method, multi-dataset benchmark)
and stays useful for exactly the case the sweep is overkill for: fit one
model on one category, right now, without a SQLite results store or a
29-category zoo. Both paths here export to the same production ONNX contract
the sweep's native methods use, and both are verified against the real
``OnnxInferenceEngine`` + ``ThresholdAnomalyDetector`` before being called done.

Organized in five parts:

1. ``ConvAutoencoder`` / ``AnomalyExportModel`` -- a from-scratch
   reconstruction autoencoder, trained with an MSE loss.
2. ``PaDiMFeatureExtractor`` / ``PaDiMModel`` -- PaDiM on a pretrained
   torchvision backbone (a simpler, single-file cousin of
   ``training.models``'s PaDiM, predating the backbone-as-a-swappable-axis
   design there).
3. ``auroc`` -- a tiny rank-based AUROC, so this module doesn't need scikit-learn.
4. The autoencoder training/export/verify CLI (``main_autoencoder``).
5. The PaDiM training/export/verify CLI (``main_padim``).

Dispatched from the unified CLI in ``training.train`` as
``python -m training.train autoencoder ...`` / ``... padim ...``.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)
from tqdm import tqdm

from adaptivevision.common import ExecutionProvider, RectifiedFrame
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.metrology import ThresholdAnomalyDetector
from training.data import DATASET_LOADERS, load_rgb

REPO_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# From-scratch reconstruction autoencoder
# -----------------------------------------------------------------------------
#
# ConvAutoencoder is trained on normal-only images with an MSE reconstruction
# loss. AnomalyExportModel wraps the trained autoencoder so the *exported
# ONNX graph itself* matches the production contract consumed by
# ThresholdAnomalyDetector: input named "input" of static shape (1, H, W)
# (channel dim only, no batch axis), output named "output" -- a single
# scalar in [0, 1].


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for single-channel square images.

    ``image_size`` must be divisible by 16 (four stride-2 downsamples).
    """

    def __init__(self, image_size: int, latent_channels: int = 128) -> None:
        super().__init__()
        if image_size % 16 != 0:
            msg = f"image_size must be divisible by 16, got {image_size}"
            raise ValueError(msg)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x`` (shape ``(N, 1, H, W)``, values in ``[0, 1]``)."""
        return self.decoder(self.encoder(x))


class AnomalyExportModel(nn.Module):
    """Wraps a trained autoencoder into the exact production ONNX contract.

    Args:
        autoencoder: Trained, frozen ``ConvAutoencoder``.
        mu: Mean reconstruction score on held-out normal validation data.
        sigma: Standard deviation of that score (floor-clamped by the caller).
        topk_fraction: Fraction of worst-reconstructed pixels averaged into
            the raw anomaly score, so localized defects aren't diluted by a
            plain whole-image MSE.
    """

    def __init__(
        self,
        autoencoder: ConvAutoencoder,
        mu: float,
        sigma: float,
        topk_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.topk_fraction = topk_fraction

    def raw_score(self, x_chw_0_1: torch.Tensor) -> torch.Tensor:
        """Top-k-fraction mean squared reconstruction error for a batch."""
        recon = self.autoencoder(x_chw_0_1)
        err = (recon - x_chw_0_1) ** 2
        err_flat = err.reshape(err.shape[0], -1)
        k = max(1, int(err_flat.shape[1] * self.topk_fraction))
        topk = torch.topk(err_flat, k, dim=1).values
        return topk.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a single frame ``x`` of shape ``(1, H, W)``, values ``[0, 255]``.

        Returns:
            A ``(1,)`` tensor with a calibrated anomaly score in ``[0, 1]``.
        """
        batched = (x / 255.0).unsqueeze(0)
        raw = self.raw_score(batched)
        score = torch.sigmoid((raw - self.mu) / self.sigma)
        return score.reshape(1)


# -----------------------------------------------------------------------------
# PaDiM on a pretrained torchvision backbone
# -----------------------------------------------------------------------------
#
# Reconstruction autoencoders trained from scratch on a few hundred images
# top out well below what's needed for a strong result. PaDiM (Defard et al.,
# 2020) instead extracts multi-layer features from an ImageNet-pretrained
# CNN, fits a per-patch-position Gaussian from the normal-only training
# images, and scores new images by Mahalanobis distance to those Gaussians.
# It needs no gradient descent at all -- fitting is a single forward pass
# over the training images plus a covariance/inverse computation.
#
# Everything here (backbone forward, channel selection, Mahalanobis distance,
# sigmoid calibration) is plain tensor ops, so the *whole* scorer -- backbone
# included -- exports to one ONNX graph matching the same production
# contract used by the autoencoder above: input "input" of static shape
# (3, H, W), output "output" -- a scalar anomaly score in [0, 1].

#: Per-backbone (layer1, layer2, layer3) channel counts, so the caller can
#: size the concatenated feature map without instantiating the backbone.
_BACKBONE_CHANNELS = {
    "resnet18": (64, 128, 256),
    "wide_resnet50_2": (256, 512, 1024),
}


class PaDiMFeatureExtractor(nn.Module):
    """Frozen backbone trunk producing a concatenated multi-layer feature map.

    ``wide_resnet50_2`` matches PaDiM's best-reported results but costs more
    compute/memory than ``resnet18``; both are ImageNet-pretrained and frozen.
    """

    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()
        if backbone_name == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == "wide_resnet50_2":
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        else:
            msg = f"Unknown backbone: {backbone_name!r}"
            raise ValueError(msg)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self._num_channels = sum(_BACKBONE_CHANNELS[backbone_name])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def num_channels(self) -> int:
        """Total channel count of the concatenated feature map."""
        return self._num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch ``x`` of shape ``(N, 3, H, W)``, values ``[0, 255]``."""
        x = (x / 255.0 - self.mean) / self.std
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        size = f1.shape[-2:]
        f2_up = nn.functional.interpolate(f2, size=size, mode="nearest")
        f3_up = nn.functional.interpolate(f3, size=size, mode="nearest")
        return torch.cat([f1, f2_up, f3_up], dim=1)


class PaDiMModel(nn.Module):
    """Wraps the extractor + fitted Gaussians into the production ONNX contract.

    Args:
        extractor: Feature extractor (frozen).
        channel_indices: Fixed random subset of feature channels, shape ``(d,)``.
        mean: Per-patch-position mean, shape ``(P, d)``.
        inv_cov: Per-patch-position inverse covariance, shape ``(P, d, d)``.
        mu: Mean raw Mahalanobis score on held-out normal validation data.
        sigma: Standard deviation of that score (floor-clamped by the caller).
    """

    def __init__(
        self,
        extractor: PaDiMFeatureExtractor,
        channel_indices: torch.Tensor,
        mean: torch.Tensor,
        inv_cov: torch.Tensor,
        mu: float,
        sigma: float,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.register_buffer("channel_indices", channel_indices)
        self.register_buffer("mean", mean)
        self.register_buffer("inv_cov", inv_cov)
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reduced-dimension per-patch feature matrix ``(P, d)`` for one image."""
        features = self.extractor(x.unsqueeze(0))
        selected = torch.index_select(features, 1, self.channel_indices)
        _, channels, height, width = selected.shape
        return selected.reshape(channels, height * width).transpose(0, 1)

    def raw_score(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """Max per-patch Mahalanobis distance over all patch positions."""
        diff = patch_feats - self.mean
        weighted = torch.bmm(diff.unsqueeze(1), self.inv_cov).squeeze(1)
        mahalanobis = (weighted * diff).sum(dim=1)
        return mahalanobis.max()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a single frame ``x`` of shape ``(3, H, W)``, values ``[0, 255]``.

        Returns:
            A ``(1,)`` tensor with a calibrated anomaly score in ``[0, 1]``.
        """
        raw = self.raw_score(self.patch_features(x))
        score = torch.sigmoid((raw - self.mu) / self.sigma)
        return score.reshape(1)


def fit_gaussians(
    extractor: PaDiMFeatureExtractor,
    features_nd: torch.Tensor,
    eps: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a per-patch-position Gaussian from stacked training features.

    Args:
        extractor: Unused directly; kept for signature symmetry/clarity.
        features_nd: Stacked reduced-dimension features, shape ``(N, P, d)``.
        eps: Diagonal regularization added before inversion.

    Returns:
        ``(mean, inv_cov)`` of shapes ``(P, d)`` and ``(P, d, d)``.
    """
    _ = extractor
    n = features_nd.shape[0]
    mean = features_nd.mean(dim=0)
    centered = (features_nd - mean).permute(1, 0, 2).contiguous()  # (P, N, d)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (n - 1)  # (P, d, d)
    d = cov.shape[-1]
    identity = torch.eye(d, device=cov.device).unsqueeze(0)
    cov = cov + eps * identity
    inv_cov = torch.linalg.inv(cov)
    return mean, inv_cov


def fit_gaussian_pooled(
    features_nd: torch.Tensor,
    eps: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a single position-agnostic Gaussian, pooling all patches together.

    Per-position modeling (:func:`fit_gaussians`) assumes patch position
    ``(i, j)`` means the same thing across training images -- true for
    centered, camera-aligned parts (MVTec/VisA/Kolektor), but false for
    arbitrary crops of continuous material (e.g. Severstal steel strips),
    where a defect can appear at any position and unrelated positions across
    images share no semantic alignment. Pooling every patch from every
    position into one Gaussian is the position-agnostic analogue.

    Returns:
        ``(mean, inv_cov)`` broadcast to per-position shapes ``(P, d)`` and
        ``(P, d, d)`` (repeating the single Gaussian ``P`` times) so the
        result is a drop-in replacement for :func:`fit_gaussians` -- same
        contract, same downstream ``PaDiMModel``.
    """
    n, p, d = features_nd.shape
    pooled = features_nd.reshape(n * p, d)
    mean = pooled.mean(dim=0)
    centered = pooled - mean
    cov = (centered.T @ centered) / (n * p - 1)
    cov = cov + eps * torch.eye(d, device=cov.device)
    inv_cov = torch.linalg.inv(cov)
    mean_p = mean.unsqueeze(0).expand(p, -1).contiguous()
    inv_cov_p = inv_cov.unsqueeze(0).expand(p, -1, -1).contiguous()
    return mean_p, inv_cov_p


# -----------------------------------------------------------------------------
# AUROC
# -----------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Args:
        scores: Anomaly scores, higher means more anomalous.
        labels: Boolean (or 0/1) ground-truth anomaly labels.

    Returns:
        AUROC in ``[0, 1]``, or ``nan`` if only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def resolve_paths(
    dataset: str, category: str | None, data_root: Path
) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Dispatch to the dataset-specific loader with its expected root.

    Shared by both training paths below (``main_autoencoder``/``main_padim``).
    """
    if dataset in ("mvtec",):
        if not category:
            msg = f"--category is required for dataset={dataset!r}"
            raise SystemExit(msg)
        return DATASET_LOADERS[dataset](data_root / "mvtec" / category)
    if dataset == "visa":
        if not category:
            msg = "--category is required for dataset='visa'"
            raise SystemExit(msg)
        return DATASET_LOADERS[dataset](data_root / "VisA_20220922", category)
    if dataset == "kolektor":
        return DATASET_LOADERS[dataset](data_root / "KolektorSDD2")
    if dataset == "severstal":
        return DATASET_LOADERS[dataset](data_root / "severstal-steel-defect-detection")
    msg = f"Unknown dataset: {dataset!r}"
    raise SystemExit(msg)


# -----------------------------------------------------------------------------
# Autoencoder: train, calibrate, export, verify -- dispatched as
# `python -m training.train autoencoder ...`
# -----------------------------------------------------------------------------


def _load_grayscale(path: Path, image_size: int) -> np.ndarray:
    """Read ``path`` as grayscale and resize to ``(image_size, image_size)``."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        msg = f"Could not read image: {path}"
        raise FileNotFoundError(msg)
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


class _GrayscaleImageDataset(Dataset):
    """Loads grayscale images from disk on demand, as ``(1, H, W)`` float32."""

    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = _load_grayscale(self.paths[index], self.image_size)
        return torch.from_numpy(image).unsqueeze(0)


def train_autoencoder(
    autoencoder: ConvAutoencoder,
    train_paths: list[Path],
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_workers: int = 4,
) -> None:
    """Train ``autoencoder`` in place with an MSE reconstruction loss."""
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        _GrayscaleImageDataset(train_paths, image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.MSELoss()

    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            batch = batch.to(device, non_blocking=use_cuda) / 255.0
            optimizer.zero_grad()
            recon = autoencoder(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        mean_loss = total_loss / len(train_paths)
        print(f"epoch {epoch + 1}/{epochs}  mean_loss={mean_loss:.5f}  lr={lr_now:.2e}")


def compute_calibration(
    autoencoder: ConvAutoencoder,
    val_paths: list[Path],
    image_size: int,
    topk_fraction: float,
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(mu, sigma)`` of raw top-k reconstruction scores on ``val_paths``."""
    scorer = AnomalyExportModel(autoencoder, mu=0.0, sigma=1.0, topk_fraction=topk_fraction)
    scorer.eval()
    scores = []
    with torch.no_grad():
        for path in val_paths:
            image = torch.from_numpy(_load_grayscale(path, image_size)).unsqueeze(0).to(device)
            batched = (image / 255.0).unsqueeze(0)
            scores.append(scorer.raw_score(batched).item())
    mu = float(np.mean(scores))
    sigma = max(float(np.std(scores)), 1e-6)
    return mu, sigma


def evaluate_autoencoder(
    autoencoder: ConvAutoencoder,
    mu: float,
    sigma: float,
    test_split: list[tuple[Path, bool]],
    image_size: int,
    topk_fraction: float,
    device: torch.device,
) -> tuple[float, list[tuple[Path, bool, float]]]:
    """Score the labeled test split and return ``(auroc, [(path, label, score)])``."""
    scorer = AnomalyExportModel(autoencoder, mu=mu, sigma=sigma, topk_fraction=topk_fraction).to(
        device
    )
    scorer.eval()
    results = []
    with torch.no_grad():
        for path, is_anomalous in test_split:
            image = torch.from_numpy(_load_grayscale(path, image_size)).to(device)
            score = scorer(image).item()
            results.append((path, is_anomalous, score))
    scores = np.array([r[2] for r in results])
    labels = np.array([r[1] for r in results])
    return auroc(scores, labels), results


def export_onnx(
    autoencoder: ConvAutoencoder,
    mu: float,
    sigma: float,
    topk_fraction: float,
    image_size: int,
    output_path: Path,
) -> None:
    """Export the calibrated anomaly scorer to a single static-shape ONNX graph."""
    export_model = AnomalyExportModel(autoencoder, mu=mu, sigma=sigma, topk_fraction=topk_fraction)
    export_model.eval()
    dummy_input = torch.zeros(1, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
        # torch>=2.9 defaults to the dynamo exporter, which needs onnxscript.
        dynamo=False,
    )


def verify_autoencoder_with_production_code(
    output_path: Path,
    test_split: list[tuple[Path, bool]],
    image_size: int,
) -> None:
    """Run the exported model through the project's real inference/detector code.

    Uses ``OnnxInferenceEngine`` and ``ThresholdAnomalyDetector`` exactly as
    the production pipeline would, proving the exported graph satisfies the
    contract end-to-end rather than just matching it on paper.
    """
    engine = OnnxInferenceEngine(model_dir=output_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(output_path.name)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)

    normal_example = next((p for p, anomalous in test_split if not anomalous), None)
    anomaly_example = next((p for p, anomalous in test_split if anomalous), None)

    for label, path in (("normal", normal_example), ("anomalous", anomaly_example)):
        if path is None:
            print(f"  (no {label} example available in test split)")
            continue
        image = _load_grayscale(path, image_size)
        frame = RectifiedFrame(
            image=image,
            camera_id="training-verify",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        result = detector.detect(frame)
        print(
            f"  {label:>10}: score={result.score:.4f} "
            f"is_anomalous={result.is_anomalous} ({path.name})"
        )


def main_autoencoder(argv: list[str] | None = None) -> None:
    """Parse CLI args, train, calibrate, export, and verify the autoencoder.

    Usage:
        python -m training.train autoencoder --dataset mvtec --category bottle
        python -m training.train autoencoder --dataset visa --category candle
        python -m training.train autoencoder --dataset kolektor
        python -m training.train autoencoder --dataset severstal
    """
    parser = argparse.ArgumentParser(description=main_autoencoder.__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LOADERS))
    parser.add_argument("--category", default=None, help="Required for mvtec/visa.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--topk-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, e.g. 'cuda' or 'cpu' (default: cuda if available).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Parallel DataLoader workers for image decoding (default: 4).",
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"training on device={device}")

    output_path = args.output
    if output_path is None:
        name = args.dataset if not args.category else f"{args.dataset}_{args.category}"
        output_path = REPO_ROOT / "models" / f"{name}.onnx"

    print(f"Loading dataset paths for dataset={args.dataset!r} category={args.category!r} ...")
    normal_paths, test_split = resolve_paths(args.dataset, args.category, args.data_root)
    if len(normal_paths) < 10:
        msg = f"Only {len(normal_paths)} normal training images found; check --data-root."
        raise SystemExit(msg)

    random.shuffle(normal_paths)
    n_val = max(1, int(len(normal_paths) * args.val_fraction))
    val_paths, train_paths = normal_paths[:n_val], normal_paths[n_val:]
    print(f"train={len(train_paths)} val={len(val_paths)} test={len(test_split)}")

    autoencoder = ConvAutoencoder(args.image_size).to(device)
    train_autoencoder(
        autoencoder,
        train_paths,
        args.image_size,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
        num_workers=args.num_workers,
    )

    mu, sigma = compute_calibration(
        autoencoder, val_paths, args.image_size, args.topk_fraction, device
    )
    print(f"calibration: mu={mu:.5f} sigma={sigma:.5f}")

    if test_split:
        score, _results = evaluate_autoencoder(
            autoencoder, mu, sigma, test_split, args.image_size, args.topk_fraction, device
        )
        print(f"test AUROC={score:.4f}  (recommended DecisionPolicy.anomaly_threshold=0.5)")

    # ONNX export runs on CPU: it matches production's CPUExecutionProvider,
    # and torch.onnx.export wants the dummy input on the same device as the model.
    autoencoder.to("cpu")
    export_onnx(autoencoder, mu, sigma, args.topk_fraction, args.image_size, output_path)
    print(f"exported ONNX model: {output_path}")

    print("verifying with production OnnxInferenceEngine + ThresholdAnomalyDetector:")
    verify_autoencoder_with_production_code(output_path, test_split, args.image_size)


# -----------------------------------------------------------------------------
# PaDiM: fit, calibrate, export, verify -- dispatched as
# `python -m training.train padim ...`
# -----------------------------------------------------------------------------

#: Native aspect ratios differ a lot across these datasets (Severstal strips
#: are ~6:1, Kolektor panels ~2.8:1); resizing everything to a square would
#: squash the very defects we're trying to detect, so each dataset gets a
#: (height, width) tuned to its real proportions. Deliberately its own table,
#: separate from training.data.DEFAULT_SIZE: this legacy path was tuned at
#: 224x224 (the standard ImageNet-pretrained-backbone convention) rather than
#: the benchmark zoo's 256x256.
_PADIM_DEFAULT_SIZE = {
    "mvtec": (224, 224),
    "visa": (224, 224),
    "kolektor": (224, 80),
    "severstal": (128, 800),
}

#: Per-position modeling assumes camera-aligned parts; Severstal crops are
#: unaligned sections of continuous steel strip, so it needs the pooled,
#: position-agnostic Gaussian instead (see :func:`fit_gaussian_pooled`).
_PADIM_DEFAULT_POOLED = {
    "mvtec": False,
    "visa": False,
    "kolektor": False,
    "severstal": True,
}


def extract_patch_features(
    extractor: PaDiMFeatureExtractor,
    channel_indices: torch.Tensor,
    paths: list[Path],
    height: int,
    width: int,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Run the frozen backbone over ``paths`` and return stacked ``(N, P, d)`` features."""
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack(
                [torch.from_numpy(load_rgb(p, height, width)) for p in batch_paths]
            ).to(device)
            features = extractor(batch)
            selected = torch.index_select(features, 1, channel_indices)
            n, channels, h, w = selected.shape
            chunks.append(selected.reshape(n, channels, h * w).permute(0, 2, 1))
    return torch.cat(chunks, dim=0)


def evaluate_padim(
    model: PaDiMModel,
    test_split: list[tuple[Path, bool]],
    height: int,
    width: int,
    device: torch.device,
) -> float:
    """Score the labeled test split with the fitted model and return AUROC."""
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for path, is_anomalous in test_split:
            image = torch.from_numpy(load_rgb(path, height, width)).to(device)
            scores.append(model(image).item())
            labels.append(is_anomalous)
    return auroc(np.array(scores), np.array(labels))


def verify_padim_with_production_code(
    output_path: Path, test_split: list[tuple[Path, bool]], height: int, width: int
) -> None:
    """Run the exported ONNX model through the real production inference/detector code."""
    engine = OnnxInferenceEngine(model_dir=output_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(output_path.name)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)

    normal_example = next((p for p, anomalous in test_split if not anomalous), None)
    anomaly_example = next((p for p, anomalous in test_split if anomalous), None)
    for label, path in (("normal", normal_example), ("anomalous", anomaly_example)):
        if path is None:
            print(f"  (no {label} example available in test split)")
            continue
        image = load_rgb(path, height, width)
        frame = RectifiedFrame(
            image=image,
            camera_id="training-verify",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        result = detector.detect(frame)
        print(
            f"  {label:>10}: score={result.score:.4f} "
            f"is_anomalous={result.is_anomalous} ({path.name})"
        )


def main_padim(argv: list[str] | None = None) -> None:
    """Parse CLI args, fit PaDiM, calibrate, export, and verify.

    Usage:
        python -m training.train padim --dataset mvtec --category bottle
        python -m training.train padim --dataset visa --category candle
        python -m training.train padim --dataset kolektor
        python -m training.train padim --dataset severstal
    """
    parser = argparse.ArgumentParser(description=main_padim.__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(_PADIM_DEFAULT_SIZE))
    parser.add_argument("--category", default=None, help="Required for mvtec/visa.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--fit-samples", type=int, default=250)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--reduced-dims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "wide_resnet50_2"])
    parser.add_argument(
        "--pooled",
        dest="pooled",
        action="store_true",
        default=None,
        help="Force a single position-agnostic Gaussian (see fit_gaussian_pooled).",
    )
    parser.add_argument(
        "--no-pooled",
        dest="pooled",
        action="store_false",
        help="Force per-patch-position Gaussians even for datasets defaulting to pooled.",
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pooled = _PADIM_DEFAULT_POOLED[args.dataset] if args.pooled is None else args.pooled
    print(f"fitting on device={device} backbone={args.backbone} pooled={pooled}")

    height, width = _PADIM_DEFAULT_SIZE[args.dataset]
    if args.height:
        height = args.height
    if args.width:
        width = args.width

    output_path = args.output
    if output_path is None:
        name = args.dataset if not args.category else f"{args.dataset}_{args.category}"
        output_path = REPO_ROOT / "models" / f"padim_{name}.onnx"

    print(f"Loading dataset paths for dataset={args.dataset!r} category={args.category!r} ...")
    normal_paths, test_split = resolve_paths(args.dataset, args.category, args.data_root)
    if len(normal_paths) < 10:
        msg = f"Only {len(normal_paths)} normal training images found; check --data-root."
        raise SystemExit(msg)

    random.shuffle(normal_paths)
    n_val = max(1, int(len(normal_paths) * args.val_fraction))
    val_paths = normal_paths[:n_val]
    fit_paths = normal_paths[n_val : n_val + args.fit_samples]
    print(
        f"fit={len(fit_paths)} val={len(val_paths)} test={len(test_split)} size=({height}x{width})"
    )

    extractor = PaDiMFeatureExtractor(args.backbone).to(device)
    extractor.eval()
    generator = torch.Generator().manual_seed(args.seed)
    channel_indices = torch.randperm(extractor.num_channels, generator=generator)[
        : args.reduced_dims
    ].to(device)

    print("extracting training features ...")
    fit_features = extract_patch_features(
        extractor, channel_indices, fit_paths, height, width, device
    )
    print(f"fitting Gaussians over {fit_features.shape[1]} patch positions (pooled={pooled}) ...")
    mean, inv_cov = (
        fit_gaussian_pooled(fit_features) if pooled else fit_gaussians(extractor, fit_features)
    )

    calibration_model = PaDiMModel(extractor, channel_indices, mean, inv_cov, mu=0.0, sigma=1.0)
    calibration_model.eval()
    val_scores = []
    with torch.no_grad():
        for path in val_paths:
            image = torch.from_numpy(load_rgb(path, height, width)).to(device)
            raw = calibration_model.raw_score(calibration_model.patch_features(image))
            val_scores.append(raw.item())
    mu = float(np.mean(val_scores))
    sigma = max(float(np.std(val_scores)), 1e-6)
    print(f"calibration: mu={mu:.5f} sigma={sigma:.5f}")

    model = PaDiMModel(extractor, channel_indices, mean, inv_cov, mu=mu, sigma=sigma)
    model.eval()

    test_auroc = None
    if test_split:
        test_auroc = evaluate_padim(model, test_split, height, width, device)
        print(f"test AUROC={test_auroc:.4f}  (recommended DecisionPolicy.anomaly_threshold=0.5)")

    model.to("cpu")
    model.eval()
    dummy_input = torch.zeros(3, height, width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
        # torch>=2.9 defaults to the dynamo exporter, which needs onnxscript.
        dynamo=False,
    )
    print(f"exported ONNX model: {output_path}")

    manifest = {
        "dataset": args.dataset,
        "category": args.category,
        "height": height,
        "width": width,
        "backbone": args.backbone,
        "pooled": pooled,
        "test_auroc": test_auroc,
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))

    print("verifying with production OnnxInferenceEngine + ThresholdAnomalyDetector:")
    verify_padim_with_production_code(output_path, test_split, height, width)
