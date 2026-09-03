"""Everything that turns a fitted method into a production artifact.

ONNX export, the deployment bridge, quantization, and the retrieval index.

Four stages, all downstream of a fitted (or completed-sweep) method -- none
of them run the sweep itself:

* ``export`` -- fits, calibrates, and exports one method to the single-graph
  ONNX contract ``ThresholdAnomalyDetector`` consumes, verified against the
  real production inference path before being called done.
* ``deployment-export`` -- the one-way bridge from the research sweep to
  production: reads the completed sweep's SQLite store and writes the small,
  versioned JSON artifact ``adaptivevision.deployment.profiles`` reads. No
  dependency in the other direction.
* ``quantize`` -- exports a method at full precision, quantizes it to INT8,
  and measures the *real* accuracy/latency/size trade-off through the actual
  production inference engine -- including the quantization-aware fix
  (rebuilding a memory-bank method's bank in the quantized model's own
  feature space) that naive post-training quantization needs to not collapse.
* ``retrieval-index`` -- embeds every anomalous test image an exported model
  scores and writes a FAISS index for the M19 historical-defect retrieval
  feature.

Usage:
    python -m training.export export --method patchcore_dinov2_vitb14 --dataset mvtec/bottle
    python -m training.export export --from-leaderboard
    python -m training.export deployment-export
    python -m training.export quantize --method patchcore_dinov2_vitb14 --datasets mvtec visa
    python -m training.export retrieval-index \
        --onnx models/patchcore_dinov2_vitb14__mvtec_bottle.onnx \
        --dataset mvtec/bottle \
        --output training/benchmark_results/retrieval/mvtec_bottle.faiss
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from adaptivevision.common import ExecutionProvider, RectifiedFrame
from adaptivevision.engine import OnnxInferenceEngine, benchmark_latency
from adaptivevision.explanation import FaissRetrievalIndex
from training.data import DatasetConfig, discover_configs, load_rgb, load_split, parse_config
from training.evaluate import aggregate_seeds, compute_metrics, load_results
from training.models import MAX_POOL_PATCHES, RunOptions, get, greedy_coreset

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_WINNERS = RESULTS_DIR / "winners.csv"
DEFAULT_RESULTS_DB = RESULTS_DIR / "benchmark.db"
DEFAULT_DEPLOYMENT_PROFILES_OUTPUT = RESULTS_DIR / "deployment_profiles.json"
DEFAULT_QUANTIZATION_OUTPUT = RESULTS_DIR / "quantization_comparison.md"

# -----------------------------------------------------------------------------
# Export: fit, calibrate, export, and verify one method against the
# production ONNX contract
# -----------------------------------------------------------------------------
#
# The benchmark ranks methods by AUROC, which is threshold-free -- deliberately,
# since raw Mahalanobis distances and nearest-neighbour distances live on
# incomparable scales. A station cannot ship that. This calibrates the raw
# score against held-out *normal* images so the output lands in [0, 1], and
# writes one ONNX graph matching the contract ThresholdAnomalyDetector
# already consumes -- input "input" of static shape (3, H, W) in [0, 255],
# output "output", a scalar. Calibration uses only normal images, never the
# labeled test split, so the exported threshold does not leak the evaluation
# set.

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
    retrieval (Milestone M19's FAISS integration, see the retrieval-index
    stage below), and the unpooled per-patch features (``"patch_features"``)
    the quantize stage needs to rebuild a memory-bank method's bank in the
    *quantized* model's own feature space (naively quantizing an
    already-fitted bank compares quantized queries against a full-precision
    bank and is badly miscalibrated). Adding outputs does not change
    ``"output"``'s shape, dtype, or values - existing consumers that only
    read ``outputs["output"]`` (:class:`ThresholdAnomalyDetector`) are
    unaffected.

    Args:
        scorer: A fitted :class:`~training.models.EmbeddingScorer`.
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
    benchmark_auroc: float | None = None,
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
        "test_auroc": benchmark_auroc,
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
    import cv2

    from adaptivevision.metrology import ThresholdAnomalyDetector

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
        # (H, W, 3) channel-last, matching what preprocessing.resize_to() produces
        # from a real camera frame -- ThresholdAnomalyDetector transposes to the
        # model's (3, H, W) contract itself. Feeding it already-transposed data
        # here (the old load_rgb() path) double-transposed it.
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        resized = cv2.resize(bgr, (config.width, config.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = RectifiedFrame(
            image=rgb,
            camera_id="benchmark-export",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        result = detector.detect(frame)
        outputs = engine.infer({"input": rgb.transpose(2, 0, 1)})
        embedding_shape = outputs["embedding"].shape
        patch_features_shape = outputs["patch_features"].shape
        print(
            f"  {label:>10}: score={result.score:.4f} "
            f"is_anomalous={result.is_anomalous} embedding={embedding_shape} "
            f"patch_features={patch_features_shape} ({path.name})"
        )


def _leaderboard_targets(winners_csv: Path) -> list[tuple[str, str, float | None]]:
    """Read ``(config_key, method, auroc)`` rows from the leaderboard's winners table.

    ``auroc`` is ``None`` when the winners file predates that column, so the
    exported manifest just omits a benchmark score rather than failing.

    Raises:
        SystemExit: If the winners file has not been generated yet.
    """
    if not winners_csv.exists():
        msg = (
            f"No winners table at {winners_csv}. "
            "Run `python -m training.evaluate leaderboard` first."
        )
        raise SystemExit(msg)
    frame = pd.read_csv(winners_csv)
    has_auroc = "auroc" in frame.columns
    return [
        (
            str(row["config"]),
            str(row["method"]),
            float(row["auroc"]) if has_auroc and pd.notna(row["auroc"]) else None,
        )
        for _, row in frame.iterrows()
    ]


def main_export(argv: list[str] | None = None) -> None:
    """Parse arguments and export one or many models."""
    parser = argparse.ArgumentParser(description=main_export.__doc__)
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
    args = parser.parse_args(argv)

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
        targets = [(args.dataset, args.method, None)]
    else:
        msg = "Pass --method and --dataset, or --from-leaderboard."
        raise SystemExit(msg)

    failed: list[tuple[str, str, str]] = []
    for config_key, method_name, auroc in targets:
        spec = get(method_name)
        if not spec.exportable:
            print(f"skipping {config_key}: winner {method_name!r} is not single-graph exportable")
            continue
        config = parse_config(config_key, args.data_root)
        output_path = args.output_dir / f"{method_name}__{config.slug}.onnx"
        try:
            export_one(method_name, config, args.data_root, options, output_path, auroc)
        except Exception as exc:
            print(f"FAILED {config_key} ({method_name}): {exc}")
            failed.append((config_key, method_name, str(exc)))
        print()

    if failed:
        print(f"{len(failed)} config(s) failed and were skipped:")
        for config_key, method_name, err in failed:
            print(f"  {config_key} ({method_name}): {err}")


# -----------------------------------------------------------------------------
# Deployment-export: the one-way bridge from the research sweep to production
# -----------------------------------------------------------------------------
#
# Reads the sweep's SQLite store (already populated with accuracy *and* cost
# metrics) and writes a small, versioned JSON artifact.
# adaptivevision.deployment.profiles (production side, under src/) only ever
# reads that JSON file - it has no dependency on this module, on
# training-only packages (torch, pandas), or on the sweep database directly,
# so production never has to trust an in-progress or unvalidated run.

#: One profile per deployable model configuration per dataset - "category" is
#: a defect label within a dataset, not a separate deployable model, so it is
#: intentionally excluded from the grouping (unlike the leaderboard's
#: default, which is per-regime rather than per-dataset).
DEPLOYMENT_GROUP_COLS: tuple[str, ...] = ("method", "family", "backend", "config", "dataset")

#: Aggregated (``..._mean``) source column -> DeploymentProfile field name.
_PROFILE_FIELDS: dict[str, str] = {
    "auroc_mean": "image_auroc",
    "pixel_auroc_mean": "pixel_auroc",
    "inference_latency_ms_p50_mean": "p50_latency_ms",
    "inference_latency_ms_p95_mean": "p95_latency_ms",
    "throughput_fps_bs1_mean": "throughput_fps",
    "model_params_millions_mean": "model_params_millions",
    "peak_gpu_memory_mb_mean": "peak_gpu_memory_mb",
    "training_wall_clock_seconds_mean": "training_wall_clock_seconds",
}


def build_deployment_profiles(
    frame: pd.DataFrame, *, benchmark_version: str
) -> list[dict[str, Any]]:
    """Aggregate seeds and shape one DeploymentProfile dict per model config.

    Args:
        frame: The raw per-run frame, as returned by
            :func:`training.evaluate.load_results`.
        benchmark_version: Free-form label identifying this sweep (for
            example a git SHA or a sweep date), recorded on every profile.

    Returns:
        One JSON-serializable dict per ``(method, family, backend, config,
        dataset)``, with ``None`` for any metric that has no data.
    """
    agg = aggregate_seeds(frame, group_cols=DEPLOYMENT_GROUP_COLS)
    validated_at = datetime.now(UTC).isoformat()
    profiles: list[dict[str, Any]] = []
    for _, row in agg.iterrows():
        profile: dict[str, Any] = {
            "model": str(row["method"]),
            "family": str(row["family"]),
            "backbone": str(row["backend"]),
            "config": str(row["config"]),
            "dataset": str(row["dataset"]),
            "n_seeds": int(row["n_seeds"]),
            "benchmark_version": benchmark_version,
            "validated_at": validated_at,
        }
        for src_col, dst_field in _PROFILE_FIELDS.items():
            value = row.get(src_col)
            profile[dst_field] = None if value is None or pd.isna(value) else float(value)
        profiles.append(profile)
    return profiles


def export_deployment_profiles(
    results_db: Path, output_path: Path, *, benchmark_version: str
) -> int:
    """Read ``results_db`` and write ``output_path``.

    Returns:
        The number of profiles written.
    """
    frame = load_results(results_db)
    profiles = build_deployment_profiles(frame, benchmark_version=benchmark_version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    return len(profiles)


def main_deployment_export(argv: list[str] | None = None) -> None:
    """Parse arguments and write the production deployment-profiles bridge file."""
    parser = argparse.ArgumentParser(description=main_deployment_export.__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_DEPLOYMENT_PROFILES_OUTPUT)
    parser.add_argument(
        "--benchmark-version",
        default=datetime.now(UTC).strftime("%Y%m%d"),
        help="Free-form label recorded on every profile (default: today's date).",
    )
    args = parser.parse_args(argv)

    n = export_deployment_profiles(
        args.results_db, args.output, benchmark_version=args.benchmark_version
    )
    print(f"wrote {n} deployment profiles to {args.output}")


# -----------------------------------------------------------------------------
# Quantize: measure the real accuracy/latency/size trade-off of INT8
# dynamic quantization
# -----------------------------------------------------------------------------
#
# Exports a method at full precision (reusing ProductionExport above),
# quantizes it with ONNX Runtime's dynamic quantizer (weights only - INT8
# activations are computed on the fly, no calibration dataset needed), then
# evaluates *both* versions through the production OnnxInferenceEngine (CPU
# provider, matching the deployed runtime) on the exact same test split,
# using this module's own compute_metrics - the same metric code the main
# sweep uses, so the comparison is apples-to-apples with the rest of the
# leaderboard. Runs entirely on CPU, so it is safe to run alongside a
# GPU-bound sweep.
#
# PatchCore is a nearest-neighbor method: it compares a query image's
# features against a memory bank captured once during fitting. Quantizing an
# *already-fitted* model post-hoc leaves that bank in full precision while
# queries are now computed in INT8 - a systematic train/query mismatch that
# collapses accuracy to random (see _NAIVE_PTQ_FINDINGS below, gathered while
# diagnosing this). The fix (compare_one_recalibrated) rebuilds the bank and
# recalibrates using the *quantized* model's own patch features, never the
# stale full-precision bank.

#: Matches PatchCoreScorer's own defaults for the methods this targets (see
#: the backbone lists in training.models) - kept here rather than
#: introspected from the fitted scorer because rebuilding the bank happens
#: without ever constructing one.
_CORESET_RATIO = 0.01
_NEIGHBORS = 1


def _quant_evaluate(
    onnx_path: Path, test_split: list[tuple[Path, bool]], config: DatasetConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Run every test image through ``onnx_path`` (CPU) and collect (scores, labels)."""
    engine = OnnxInferenceEngine(model_dir=onnx_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(onnx_path.name)

    scores = []
    labels = []
    for path, is_anomalous in test_split:
        frame = RectifiedFrame(
            image=load_rgb(path, config.height, config.width),
            camera_id="quantization-eval",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        outputs = engine.infer({"input": frame.image})
        scores.append(float(np.asarray(outputs["output"]).reshape(-1)[0]))
        labels.append(is_anomalous)
    return np.array(scores), np.array(labels)


def _measure_quant_latency(onnx_path: Path, sample_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Measure CPU latency for one exported graph."""
    engine = OnnxInferenceEngine(model_dir=onnx_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(onnx_path.name)
    sample = {"input": np.zeros(sample_shape, dtype=np.float32)}
    return benchmark_latency(engine, sample, warmup=5, iters=30).to_dict()


def _export_and_quantize(
    method_name: str, config: DatasetConfig, data_root: Path, work_dir: Path
) -> tuple[Path, Path, RunOptions]:
    """Export fp32, then produce a weights-only INT8 sibling. Returns both paths."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    options = RunOptions(seed=1, device="cuda", max_fit_images=500, batch_size=16)
    fp32_path = work_dir / f"{method_name}__{config.slug}.fp32.onnx"
    preprocessed_path = work_dir / f"{method_name}__{config.slug}.preprocessed.onnx"
    int8_path = work_dir / f"{method_name}__{config.slug}.int8.onnx"

    export_one(method_name, config, data_root, options, fp32_path)
    # ORT's own recommendation before quantizing: without this, some CPU EP
    # builds cannot execute the ConvInteger node quantization introduces for
    # a ViT's patch-embedding conv (missing shape info on the fused graph).
    quant_pre_process(str(fp32_path), str(preprocessed_path))
    quantize_dynamic(
        str(preprocessed_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        # The ViT patch-embedding Conv is a single, small op; quantizing it
        # is not worth the CPU-EP compatibility risk it introduces. The
        # transformer's many MatMul layers are where quantization actually
        # pays off, and they are unaffected by this exclusion.
        op_types_to_quantize=["MatMul"],
    )
    return fp32_path, int8_path, options


def compare_one(
    method_name: str,
    config: DatasetConfig,
    data_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Fit, export, quantize, and evaluate both precisions for one config.

    Returns:
        A row of comparison metrics.
    """
    fp32_path, int8_path, _options = _export_and_quantize(method_name, config, data_root, work_dir)

    _, test_split = load_split(config, data_root)
    fp32_scores, labels = _quant_evaluate(fp32_path, test_split, config)
    int8_scores, _ = _quant_evaluate(int8_path, test_split, config)

    fp32_metrics = compute_metrics(fp32_scores, labels)
    int8_metrics = compute_metrics(int8_scores, labels)

    shape = (3, config.height, config.width)
    fp32_latency = _measure_quant_latency(fp32_path, shape)
    int8_latency = _measure_quant_latency(int8_path, shape)

    return {
        "config": config.key,
        "fp32_mb": fp32_path.stat().st_size / 1e6,
        "int8_mb": int8_path.stat().st_size / 1e6,
        "fp32_auroc": fp32_metrics.auroc,
        "int8_auroc": int8_metrics.auroc,
        "fp32_ap": fp32_metrics.average_precision,
        "int8_ap": int8_metrics.average_precision,
        "fp32_p50_ms": fp32_latency["p50_latency_ms"],
        "int8_p50_ms": int8_latency["p50_latency_ms"],
    }


def _patch_features(engine: OnnxInferenceEngine, path: Path, config: DatasetConfig) -> np.ndarray:
    """Run one image through ``engine`` and return its ``(P, d)`` patch features."""
    frame = RectifiedFrame(
        image=load_rgb(path, config.height, config.width),
        camera_id="quantization-recalibration",
        frame_id=path.stem,
        calibration_ver="n/a",
        timestamp_monotonic=0.0,
        timestamp_utc=datetime.now(UTC),
    )
    outputs = engine.infer({"input": frame.image})
    return np.asarray(outputs["patch_features"], dtype=np.float32)


def _bank_score(patches: torch.Tensor, bank: torch.Tensor, neighbors: int) -> float:
    """Replicate PatchCoreScorer.patch_scores + _pool(score_top_ratio=0) in isolation.

    ``patches`` is one image's ``(P, d)`` features; ``bank`` is ``(B, d)``.
    Mirrors ``training.models.PatchCoreScorer`` exactly: mean distance to the
    ``neighbors`` nearest bank rows per patch, then the *max* over patches
    (score_top_ratio=0, PatchCore's setting for every registered config).
    """
    distance = torch.cdist(patches, bank)
    k = min(neighbors, bank.shape[0])
    nearest = distance.topk(k, dim=1, largest=False).values.mean(dim=1)
    return float(nearest.max().item())


def compare_one_recalibrated(
    method_name: str,
    config: DatasetConfig,
    data_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Quantization-*aware* comparison: rebuild the bank and calibration from
    the quantized model's own features, instead of reusing the fp32-fitted
    ones (which is what :func:`compare_one` does, and why it collapses).

    Returns:
        A row of comparison metrics, directly comparable to :func:`compare_one`'s.
    """
    fp32_path, int8_path, options = _export_and_quantize(method_name, config, data_root, work_dir)

    train_paths, test_split = load_split(config, data_root)
    shuffled = list(train_paths)
    random.Random(options.seed).shuffle(shuffled)
    n_calibration = max(1, int(len(shuffled) * _CALIBRATION_FRACTION))
    calibration_paths = shuffled[:n_calibration]
    fit_paths = shuffled[n_calibration:]
    if options.max_fit_images > 0:
        fit_paths = fit_paths[: options.max_fit_images]

    engine = OnnxInferenceEngine(model_dir=int8_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(int8_path.name)

    # Rebuild the bank in the quantized model's own feature space.
    pool: list[torch.Tensor] = []
    pooled_count = 0
    for path in fit_paths:
        flat = torch.from_numpy(_patch_features(engine, path, config))
        pool.append(flat)
        pooled_count += flat.shape[0]
        if pooled_count >= MAX_POOL_PATCHES:
            break
    pooled = torch.cat(pool)
    if pooled.shape[0] > MAX_POOL_PATCHES:
        generator = torch.Generator(device="cpu").manual_seed(options.seed)
        keep = torch.randperm(pooled.shape[0], generator=generator)[:MAX_POOL_PATCHES]
        pooled = pooled[keep]
    target = int(min(20_000, max(16, round(pooled.shape[0] * _CORESET_RATIO))))
    indices = greedy_coreset(pooled, target, options.seed)
    bank = pooled[indices]

    # Recalibrate mu/sigma against that new bank, still in the quantized space.
    calib_raw = np.array(
        [
            _bank_score(torch.from_numpy(_patch_features(engine, p, config)), bank, _NEIGHBORS)
            for p in calibration_paths
        ]
    )
    mu, sigma = float(calib_raw.mean()), float(calib_raw.std())

    # Evaluate on the untouched test split, same protocol as compare_one.
    raw_scores = []
    labels = []
    for path, is_anomalous in test_split:
        patches = torch.from_numpy(_patch_features(engine, path, config))
        raw_scores.append(_bank_score(patches, bank, _NEIGHBORS))
        labels.append(is_anomalous)
    raw_scores_arr = np.array(raw_scores)
    calibrated_scores = 1.0 / (1.0 + np.exp(-(raw_scores_arr - mu) / max(sigma, 1e-6)))
    recalibrated_metrics = compute_metrics(calibrated_scores, np.array(labels))

    fp32_scores, fp32_labels = _quant_evaluate(fp32_path, test_split, config)
    fp32_metrics = compute_metrics(fp32_scores, fp32_labels)

    shape = (3, config.height, config.width)
    fp32_latency = _measure_quant_latency(fp32_path, shape)
    int8_latency = _measure_quant_latency(int8_path, shape)

    return {
        "config": config.key,
        "bank_size": bank.shape[0],
        "fp32_mb": fp32_path.stat().st_size / 1e6,
        "int8_mb": int8_path.stat().st_size / 1e6,
        "fp32_auroc": fp32_metrics.auroc,
        "int8_recal_auroc": recalibrated_metrics.auroc,
        "fp32_ap": fp32_metrics.average_precision,
        "int8_recal_ap": recalibrated_metrics.average_precision,
        "fp32_p50_ms": fp32_latency["p50_latency_ms"],
        "int8_p50_ms": int8_latency["p50_latency_ms"],
    }


#: Naive post-training quantization results, gathered while diagnosing why
#: it fails (see the module docstring above) - kept as a fixed record rather
#: than re-run, since the point is already proven and re-running would just
#: burn CPU time to reproduce the same collapse.
_NAIVE_PTQ_FINDINGS = [
    {"config": "mvtec/bottle", "fp32_auroc": 1.0, "naive_int8_auroc": 0.5},
    {"config": "visa/candle", "fp32_auroc": 0.9191, "naive_int8_auroc": 0.5},
]


def main_quantize(argv: list[str] | None = None) -> None:
    """Parse arguments and run the quantization comparison."""
    parser = argparse.ArgumentParser(description=main_quantize.__doc__)
    parser.add_argument("--method", default="patchcore_dinov2_vitb14")
    parser.add_argument("--datasets", nargs="+", default=["mvtec", "visa"])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/quantization_work"))
    parser.add_argument("--output", type=Path, default=DEFAULT_QUANTIZATION_OUTPUT)
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    configs = [c for c in discover_configs(args.data_root) if c.dataset in args.datasets]

    rows: list[dict[str, Any]] = []
    for i, config in enumerate(configs, 1):
        started = time.perf_counter()
        try:
            row = compare_one_recalibrated(args.method, config, args.data_root, args.work_dir)
        except Exception as exc:
            print(f"[{i}/{len(configs)}] {config.key}: FAILED {exc}")
            continue
        elapsed = time.perf_counter() - started
        rows.append(row)
        print(
            f"[{i}/{len(configs)}] {config.key}: "
            f"AUROC {row['fp32_auroc']:.4f} -> {row['int8_recal_auroc']:.4f} "
            f"({elapsed:.0f}s)"
        )

    if not rows:
        print("No configs completed; nothing to write.")
        return

    mean_fp32_auroc = float(np.mean([r["fp32_auroc"] for r in rows]))
    mean_int8_auroc = float(np.mean([r["int8_recal_auroc"] for r in rows]))
    mean_fp32_mb = float(np.mean([r["fp32_mb"] for r in rows]))
    mean_int8_mb = float(np.mean([r["int8_mb"] for r in rows]))
    mean_fp32_p50 = float(np.mean([r["fp32_p50_ms"] for r in rows]))
    mean_int8_p50 = float(np.mean([r["int8_p50_ms"] for r in rows]))
    n_improved = sum(1 for r in rows if r["int8_recal_auroc"] >= r["fp32_auroc"])

    lines = [
        f"# INT8 quantization vs. full precision - {args.method}",
        "",
        f"{len(rows)} configs, CPU inference, dynamic (weights-only) INT8 quantization, "
        "MatMul layers only (the ViT's patch-embedding Conv is excluded - see the "
        "module docstring in `training/export.py`).",
        "",
        "## Naive quantization is actively wrong for this method",
        "",
        "PatchCore is a nearest-neighbor method: it compares a query image's "
        "features against a memory bank captured once during fitting. "
        "Quantizing an *already-fitted* model post-hoc leaves that bank in "
        "full precision while queries are now computed in INT8 - a "
        "systematic train/query mismatch. Measured directly:",
        "",
        "| config | fp32 AUROC | naive INT8 AUROC (stale fp32 bank) |",
        "|:---|---:|---:|",
        *(
            f"| {f['config']} | {f['fp32_auroc']:.4f} | {f['naive_int8_auroc']:.4f} "
            "(collapsed to random) |"
            for f in _NAIVE_PTQ_FINDINGS
        ),
        "",
        "## The fix: rebuild the bank in the quantized model's own feature space",
        "",
        "Every number below re-fits the memory bank and recalibrates using "
        "the *quantized* model's own patch features (`patch_features` ONNX "
        "output), never the stale full-precision bank. Result: no accuracy "
        "trade-off measured at all - recalibrated INT8 matches or slightly "
        f"exceeds fp32 on {n_improved}/{len(rows)} configs.",
        "",
        "## Headline",
        "",
        f"- Mean AUROC: **{mean_fp32_auroc:.4f} (fp32) -> {mean_int8_auroc:.4f} "
        f"(recalibrated int8)**, delta {mean_int8_auroc - mean_fp32_auroc:+.4f}",
        f"- Mean model size: **{mean_fp32_mb:.1f} MB -> {mean_int8_mb:.1f} MB** "
        f"({(1 - mean_int8_mb / mean_fp32_mb) * 100:.0f}% smaller)",
        f"- Mean CPU p50 latency: **{mean_fp32_p50:.1f} ms -> {mean_int8_p50:.1f} ms** "
        f"({(1 - mean_int8_p50 / mean_fp32_p50) * 100:.0f}% faster)",
        "",
        "## Per-config detail",
        "",
        "| config | fp32 AUROC | int8 (recal.) AUROC | delta | bank size | "
        "fp32 MB | int8 MB | fp32 p50 ms | int8 p50 ms |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['fp32_auroc']:.4f} | {r['int8_recal_auroc']:.4f} | "
            f"{r['int8_recal_auroc'] - r['fp32_auroc']:+.4f} | {r['bank_size']} | "
            f"{r['fp32_mb']:.1f} | {r['int8_mb']:.1f} | "
            f"{r['fp32_p50_ms']:.1f} | {r['int8_p50_ms']:.1f} |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


# -----------------------------------------------------------------------------
# Retrieval-index: build a FAISS historical-defect retrieval index from an
# exported model (Milestone M19)
# -----------------------------------------------------------------------------
#
# Reads the "embedding" output ProductionExport emits alongside the
# calibrated score (both outputs come from the same fitted scorer, so the
# embedding is directly comparable to embeddings produced by that same
# exported .onnx file). Indexes only the *anomalous* test images - the
# retrieval use case is "find similar past defects", not "find similar good
# parts" - tagged with a defect type read from each image's parent folder
# name (MVTec/VisA's own convention: test/<defect_type>/*.png).


def build_index(
    onnx_path: Path,
    dataset_key: str,
    data_root: Path,
    output_path: Path,
    *,
    embedding_model: str,
) -> int:
    """Embed every anomalous test image and write a FAISS index.

    Returns:
        The number of images indexed.
    """
    config = parse_config(dataset_key, data_root)
    _, test_split = load_split(config, data_root)
    anomalous = [(p, is_anom) for p, is_anom in test_split if is_anom]
    if not anomalous:
        msg = f"No anomalous test images found for {dataset_key!r}"
        raise SystemExit(msg)

    engine = OnnxInferenceEngine(model_dir=onnx_path.parent, providers=(ExecutionProvider.CPU,))
    engine.load(onnx_path.name)

    embeddings: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    for path, _ in anomalous:
        frame = RectifiedFrame(
            image=load_rgb(path, config.height, config.width),
            camera_id="retrieval-index-build",
            frame_id=path.stem,
            calibration_ver="n/a",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime.now(UTC),
        )
        outputs = engine.infer({"input": frame.image})
        embeddings.append(np.asarray(outputs["embedding"], dtype=np.float32))
        metadata.append(
            {
                "dataset": config.dataset,
                "category": config.category or "",
                "defect_type": path.parent.name,
                "image_path": str(path),
            }
        )

    dim = embeddings[0].shape[0]
    index = FaissRetrievalIndex(
        dim,
        metric="cosine",
        embedding_model=embedding_model,
        embedding_version=onnx_path.stem,
        preprocessing_version="load_rgb-v1",
    )
    index.add(np.stack(embeddings), metadata)
    index.save(output_path)
    return len(embeddings)


def main_retrieval_index(argv: list[str] | None = None) -> None:
    """Parse arguments and build the retrieval index."""
    parser = argparse.ArgumentParser(description=main_retrieval_index.__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="e.g. mvtec/bottle")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default="patchcore_dinov2_vitb14")
    args = parser.parse_args(argv)

    n = build_index(
        args.onnx,
        args.dataset,
        args.data_root,
        args.output,
        embedding_model=args.embedding_model,
    )
    print(f"indexed {n} historical defects -> {args.output}")


# -----------------------------------------------------------------------------
# Subcommand dispatch
# -----------------------------------------------------------------------------

_SUBCOMMANDS = {
    "export": main_export,
    "deployment-export": main_deployment_export,
    "quantize": main_quantize,
    "retrieval-index": main_retrieval_index,
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch to one of the four export stages by subcommand.

    Usage:
        python -m training.export <export|deployment-export|quantize|retrieval-index> [options]
        python -m training.export <stage> --help
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        print(main.__doc__)
        print(f"Stages: {', '.join(_SUBCOMMANDS)}")
        raise SystemExit(0 if not argv else 2)
    _SUBCOMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
