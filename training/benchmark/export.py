"""Export a fitted native method to the production ONNX contract.

The benchmark ranks methods by AUROC, which is threshold-free -- deliberately,
since raw Mahalanobis distances and nearest-neighbour distances live on
incomparable scales. A station cannot ship that. This module closes the gap:
it fits the chosen method, calibrates its raw score against held-out *normal*
images so the output lands in ``[0, 1]``, and writes one ONNX graph matching
the contract ``ThresholdAnomalyDetector`` already consumes -- input ``"input"``
of static shape ``(3, H, W)`` in ``[0, 255]``, output ``"output"``, a scalar.

Calibration uses only normal images, never the labeled test split, so the
exported threshold does not leak the evaluation set.

Usage:
    python training/benchmark/export.py --method patchcore_dinov2_vitb14 --dataset mvtec/bottle
    python training/benchmark/export.py --from-leaderboard
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):  # Allow `python training/benchmark/export.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark.methods_native  # noqa: F401  (registers the native zoo)
from adaptivevision.common.enums import ExecutionProvider
from adaptivevision.common.types import RectifiedFrame
from adaptivevision.inference.onnx import OnnxInferenceEngine
from adaptivevision.inspection.anomaly.detector import ThresholdAnomalyDetector
from benchmark.data import DatasetConfig, load_split, parse_config
from benchmark.registry import RunOptions, get

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WINNERS = REPO_ROOT / "training" / "benchmark_results" / "winners.csv"

#: Fraction of the normal training images held out purely for calibration.
_CALIBRATION_FRACTION = 0.15

#: Floor on the calibration standard deviation. A perfectly uniform normal set
#: would otherwise divide by ~0 and saturate the sigmoid to a step function.
_SIGMA_FLOOR = 1e-6


class ProductionExport(nn.Module):
    """Wraps a fitted scorer into the station's single-graph ONNX contract.

    Emits three outputs from one graph: the calibrated score the station has
    always consumed (``"output"``), a fixed-length image embedding
    (``"embedding"``, mean-pooled over the patch grid) for historical-defect
    retrieval (Milestone M19's FAISS integration), and the unpooled
    per-patch features (``"patch_features"``) a quantization-aware pipeline
    needs to rebuild a memory-bank method's bank in the *quantized* model's
    own feature space (naively quantizing an already-fitted bank compares
    quantized queries against a full-precision bank and is badly miscalibrated
    - see ``quantize_and_compare.py``). Adding outputs does not change
    ``"output"``'s shape, dtype, or values - existing consumers that only
    read ``outputs["output"]`` (:class:`ThresholdAnomalyDetector`) are
    unaffected.

    Args:
        scorer: A fitted :class:`~benchmark.methods_native.EmbeddingScorer`.
        mu: Mean raw score over held-out normal images.
        sigma: Standard deviation of that raw score.
    """

    def __init__(self, scorer: nn.Module, mu: float, sigma: float) -> None:
        """Store the scorer and its calibration constants."""
        super().__init__()
        self.scorer = scorer
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sigma", torch.tensor(float(max(sigma, _SIGMA_FLOOR))))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score one frame ``x`` of shape ``(3, H, W)`` with values in ``[0, 255]``.

        Returns:
            ``(score, embedding, patch_features)``: a ``(1,)`` tensor holding
            a calibrated anomaly score in ``[0, 1]``; a ``(d,)`` tensor
            holding the image's pooled embedding; and a ``(P, d)`` tensor
            holding the unpooled per-patch features, all in the fitted
            scorer's own feature space.
        """
        patches = self.scorer.embed(x.unsqueeze(0))
        raw = self.scorer.score_patches(patches)
        score = torch.sigmoid((raw - self.mu) / self.sigma).reshape(1)
        embedding = patches.mean(dim=1).reshape(-1)
        patch_features = patches.reshape(patches.shape[1], patches.shape[2])
        return score, embedding, patch_features


def calibrate(scorer: nn.Module, paths: list[Path]) -> tuple[float, float]:
    """Measure the raw-score distribution of held-out normal images.

    Args:
        scorer: The fitted scorer.
        paths: Normal images excluded from the fit.

    Returns:
        ``(mu, sigma)`` of the raw scores.
    """
    with torch.no_grad():
        raw = scorer.score(paths)
    return float(np.mean(raw)), float(np.std(raw))


def export_one(
    method_name: str,
    config: DatasetConfig,
    data_root: Path,
    options: RunOptions,
    output_path: Path,
) -> dict[str, object]:
    """Fit, calibrate, export and verify one method on one configuration.

    Returns:
        The manifest written alongside the ``.onnx`` file.

    Raises:
        SystemExit: If the method is not exportable as a single graph.
    """
    spec = get(method_name)
    if not spec.exportable:
        msg = (
            f"{method_name!r} is not exportable to the single-graph contract. "
            "Anomalib-backed methods export through Anomalib's own exporter."
        )
        raise SystemExit(msg)

    train_paths, test_split = load_split(config, data_root)
    shuffled = list(train_paths)
    random.Random(options.seed).shuffle(shuffled)

    n_calibration = max(1, int(len(shuffled) * _CALIBRATION_FRACTION))
    calibration_paths = shuffled[:n_calibration]
    fit_paths = shuffled[n_calibration:]
    if options.max_fit_images > 0:
        fit_paths = fit_paths[: options.max_fit_images]

    print(
        f"fitting {method_name} on {config.key}: "
        f"fit={len(fit_paths)} calib={len(calibration_paths)}"
    )
    scorer = spec.fit(config, fit_paths, test_split, options)

    mu, sigma = calibrate(scorer, calibration_paths)
    print(f"calibration: mu={mu:.5f} sigma={sigma:.5f}")

    model = ProductionExport(scorer, mu, sigma).to("cpu").eval()
    scorer.device = torch.device("cpu")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(3, config.height, config.width)
    with torch.no_grad():
        embedding_dim = int(model(dummy)[1].shape[0])
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output", "embedding", "patch_features"],
        opset_version=17,
        dynamic_axes=None,
        # torch>=2.9 defaults to the dynamo exporter, which needs onnxscript.
        dynamo=False,
    )
    print(f"exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    manifest: dict[str, object] = {
        "method": method_name,
        "family": spec.family,
        "dataset": config.dataset,
        "category": config.category,
        "height": config.height,
        "width": config.width,
        "calibration_mu": mu,
        "calibration_sigma": sigma,
        "n_fit_images": len(fit_paths),
        "n_calibration_images": len(calibration_paths),
        "recommended_threshold": 0.5,
        "embedding_dim": embedding_dim,
        "exported_utc": datetime.now(UTC).isoformat(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _verify(output_path, test_split, config)
    return manifest


def _verify(output_path: Path, test_split: list[tuple[Path, bool]], config: DatasetConfig) -> None:
    """Run the exported graph through the real production inference path.

    Exporting a graph that loads is not the same as exporting a graph the
    station can use, so this drives it through ``OnnxInferenceEngine`` and
    ``ThresholdAnomalyDetector`` exactly as the pipeline would.
    """
    from image_io import load_rgb

    print("verifying with production OnnxInferenceEngine + ThresholdAnomalyDetector:")
    engine = OnnxInferenceEngine(model_dir=output_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(output_path.name)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)

    examples = [
        ("normal", next((p for p, anomalous in test_split if not anomalous), None)),
        ("anomalous", next((p for p, anomalous in test_split if anomalous), None)),
    ]
    for label, path in examples:
        if path is None:
            print(f"  (no {label} example in test split)")
            continue
        frame = RectifiedFrame(
            image=load_rgb(path, config.height, config.width),
            camera_id="benchmark-export",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        result = detector.detect(frame)
        outputs = engine.infer({"input": frame.image})
        embedding_shape = outputs["embedding"].shape
        patch_features_shape = outputs["patch_features"].shape
        print(
            f"  {label:>10}: score={result.score:.4f} "
            f"is_anomalous={result.is_anomalous} embedding={embedding_shape} "
            f"patch_features={patch_features_shape} ({path.name})"
        )


def _leaderboard_targets(winners_csv: Path) -> list[tuple[str, str]]:
    """Read ``(config_key, method)`` pairs from the leaderboard's winners table.

    Raises:
        SystemExit: If the winners file has not been generated yet.
    """
    import pandas as pd

    if not winners_csv.exists():
        msg = f"No winners table at {winners_csv}. Run training/benchmark/leaderboard.py first."
        raise SystemExit(msg)
    frame = pd.read_csv(winners_csv)
    return [(str(row["config"]), str(row["method"])) for _, row in frame.iterrows()]


def main() -> None:
    """Parse arguments and export one or many models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=None, help="Method name from the zoo.")
    parser.add_argument("--dataset", default=None, help="Config key, e.g. mvtec/bottle.")
    parser.add_argument(
        "--from-leaderboard",
        action="store_true",
        help="Export the winning exportable method for every configuration.",
    )
    parser.add_argument("--winners", type=Path, default=DEFAULT_WINNERS)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-fit-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    options = RunOptions(
        device=args.device,
        max_fit_images=args.max_fit_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.from_leaderboard:
        targets = _leaderboard_targets(args.winners)
    elif args.method and args.dataset:
        targets = [(args.dataset, args.method)]
    else:
        msg = "Pass --method and --dataset, or --from-leaderboard."
        raise SystemExit(msg)

    for config_key, method_name in targets:
        spec = get(method_name)
        if not spec.exportable:
            print(f"skipping {config_key}: winner {method_name!r} is not single-graph exportable")
            continue
        config = parse_config(config_key, args.data_root)
        output_path = args.output_dir / f"{method_name}__{config.slug}.onnx"
        export_one(method_name, config, args.data_root, options, output_path)
        print()


if __name__ == "__main__":
    main()
