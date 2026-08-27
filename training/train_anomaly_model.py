"""Train a per-category reconstruction-error anomaly autoencoder and export ONNX.

Trains on normal-only images (no defect labels needed for training), then
exports a single ONNX graph that matches the production contract consumed by
``adaptivevision.inspection.anomaly.detector.ThresholdAnomalyDetector``:
input ``"input"`` of static shape ``(1, H, W)``, output ``"output"`` -- a
scalar anomaly score already calibrated into ``[0, 1]``.

Usage:
    python training/train_anomaly_model.py --dataset mvtec --category bottle
    python training/train_anomaly_model.py --dataset visa --category candle
    python training/train_anomaly_model.py --dataset kolektor
    python training/train_anomaly_model.py --dataset severstal
"""

from __future__ import annotations

import argparse
import random
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from datasets import DATASET_LOADERS
from metrics import auroc
from model import AnomalyExportModel, ConvAutoencoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from adaptivevision.common.enums import ExecutionProvider
from adaptivevision.common.types import RectifiedFrame
from adaptivevision.inference.onnx import OnnxInferenceEngine
from adaptivevision.inspection.anomaly.detector import ThresholdAnomalyDetector

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_grayscale(path: Path, image_size: int) -> np.ndarray:
    """Read ``path`` as grayscale and resize to ``(image_size, image_size)``."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        msg = f"Could not read image: {path}"
        raise FileNotFoundError(msg)
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


class ImagePathDataset(Dataset):
    """Loads grayscale images from disk on demand, as ``(1, H, W)`` float32."""

    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = load_grayscale(self.paths[index], self.image_size)
        return torch.from_numpy(image).unsqueeze(0)


def resolve_paths(
    dataset: str, category: str | None, data_root: Path
) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Dispatch to the dataset-specific loader with its expected root."""
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
        ImagePathDataset(train_paths, image_size),
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
            image = torch.from_numpy(load_grayscale(path, image_size)).unsqueeze(0).to(device)
            batched = (image / 255.0).unsqueeze(0)
            scores.append(scorer.raw_score(batched).item())
    mu = float(np.mean(scores))
    sigma = max(float(np.std(scores)), 1e-6)
    return mu, sigma


def evaluate(
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
            image = torch.from_numpy(load_grayscale(path, image_size)).to(device)
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


def verify_with_production_code(
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
        image = load_grayscale(path, image_size)
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


def main() -> None:
    """Parse CLI args, train, calibrate, export, and verify."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

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
        score, _results = evaluate(
            autoencoder, mu, sigma, test_split, args.image_size, args.topk_fraction, device
        )
        print(f"test AUROC={score:.4f}  (recommended DecisionPolicy.anomaly_threshold=0.5)")

    # ONNX export runs on CPU: it matches production's CPUExecutionProvider,
    # and torch.onnx.export wants the dummy input on the same device as the model.
    autoencoder.to("cpu")
    export_onnx(autoencoder, mu, sigma, args.topk_fraction, args.image_size, output_path)
    print(f"exported ONNX model: {output_path}")

    print("verifying with production OnnxInferenceEngine + ThresholdAnomalyDetector:")
    verify_with_production_code(output_path, test_split, args.image_size)


if __name__ == "__main__":
    main()
