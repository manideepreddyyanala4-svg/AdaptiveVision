"""Fit a PaDiM anomaly model on real defect datasets and export it to ONNX.

Unlike the from-scratch autoencoder (``train_anomaly_model.py``), this needs
no epochs: it extracts ImageNet-pretrained features from the normal-only
training images (one forward pass, no gradients) and fits a per-patch-position
Gaussian. See ``padim.py`` for the method itself.

Usage:
    python training/train_padim.py --dataset mvtec --category bottle
    python training/train_padim.py --dataset visa --category candle
    python training/train_padim.py --dataset kolektor
    python training/train_padim.py --dataset severstal
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from image_io import load_rgb
from metrics import auroc
from padim import PaDiMFeatureExtractor, PaDiMModel, fit_gaussian_pooled, fit_gaussians
from train_anomaly_model import resolve_paths

from adaptivevision.common.enums import ExecutionProvider
from adaptivevision.common.types import RectifiedFrame
from adaptivevision.inference.onnx import OnnxInferenceEngine
from adaptivevision.inspection.anomaly.detector import ThresholdAnomalyDetector

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Native aspect ratios differ a lot across these datasets (Severstal strips
#: are ~6:1, Kolektor panels ~2.8:1); resizing everything to a square would
#: squash the very defects we're trying to detect, so each dataset gets a
#: (height, width) tuned to its real proportions.
DEFAULT_SIZE = {
    "mvtec": (224, 224),
    "visa": (224, 224),
    "kolektor": (224, 80),
    "severstal": (128, 800),
}

#: Per-position modeling assumes camera-aligned parts; Severstal crops are
#: unaligned sections of continuous steel strip, so it needs the pooled,
#: position-agnostic Gaussian instead (see ``padim.fit_gaussian_pooled``).
DEFAULT_POOLED = {
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


def evaluate(
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


def verify_with_production_code(
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


def main() -> None:
    """Parse CLI args, fit PaDiM, calibrate, export, and verify."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_SIZE))
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
        help="Force a single position-agnostic Gaussian (see padim.fit_gaussian_pooled).",
    )
    parser.add_argument(
        "--no-pooled",
        dest="pooled",
        action="store_false",
        help="Force per-patch-position Gaussians even for datasets defaulting to pooled.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pooled = DEFAULT_POOLED[args.dataset] if args.pooled is None else args.pooled
    print(f"fitting on device={device} backbone={args.backbone} pooled={pooled}")

    height, width = DEFAULT_SIZE[args.dataset]
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
        test_auroc = evaluate(model, test_split, height, width, device)
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
    verify_with_production_code(output_path, test_split, height, width)


if __name__ == "__main__":
    main()
