"""Everything that scores a completed sweep: metrics, cost, ensembling, and the leaderboard.

Five stages, all read-mostly against what ``training.sweep`` already wrote
(the SQLite store and the per-run artifact/checkpoint files from
``training.store``) -- none of them re-fit a model:

* ``cost`` -- loads each run's saved checkpoint and times its forward pass,
  backfilling latency/throughput/params/VRAM onto the same row.
* ``deploy`` -- a separate, from-scratch multi-device (CPU *and* GPU)
  latency/size profile for a short list of top methods, not backfilled onto
  the sweep table (see :func:`profile_method`'s docstring for how this
  differs from ``cost``).
* ``metrics`` -- backfills PG2/PB2/AUPIMO from the sweep's stored ``.npz``
  artifacts, for any row that predates those metrics.
* ``ensemble`` -- fuses methods from the stored artifacts. No GPU, seconds.
- ``leaderboard`` -- aggregates the sweep into a ranking and a written report.

Usage:
    python -m training.evaluate cost --results-db training/benchmark_results/benchmark.db
    python -m training.evaluate deploy --from-leaderboard --dataset mvtec/bottle
    python -m training.evaluate metrics --results-db training/benchmark_results/benchmark.db
    python -m training.evaluate ensemble --regime multiclass --max-members 3
    python -m training.evaluate leaderboard
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from training import store
from training.data import DatasetConfig, has_masks, load_mask, pixel_sample_indices, subsample_fit
from training.models import RunOptions, get

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_RESULTS_DB = RESULTS_DIR / "benchmark.db"
DEFAULT_ARTIFACTS = RESULTS_DIR / "artifacts"
DEFAULT_CHECKPOINTS = RESULTS_DIR / "checkpoints"

# -----------------------------------------------------------------------------
# Metrics shared by every method in the zoo
# -----------------------------------------------------------------------------
#
# AUROC ranks models but does not tell you what a line would actually cost. A
# station has to commit to one threshold, and the two errors it can make have
# completely different prices: scrapping a good part wastes material,
# shipping a defective one reaches a customer. So alongside the ranking
# metrics this reports both error rates at fixed, interpretable operating
# points: FPR@95TPR (the over-rejection/scrap rate if you insist on catching
# 95% of defects) and FNR@1FPR (the escape rate if you can only tolerate 1%
# false alarms).
#
# Localization is scored separately where ground-truth masks exist. Pixel
# AUROC alone flatters a model that lights up a large blob near the defect,
# so AUPRO (per-region overlap, integrated to 30% FPR) is reported with it --
# it weights every defect region equally regardless of size, which is what
# matters when the small defects are the expensive ones.

#: Recall the line is required to hit when quoting a scrap rate.
_TARGET_RECALL = 0.95

#: False-alarm budget the line is allowed when quoting an escape rate.
_FALSE_ALARM_BUDGET = 0.01

#: AUPRO integrates the PRO curve up to this false-positive rate, per the
#: MVTec AD protocol. Beyond it the operating points are useless anyway.
_AUPRO_FPR_LIMIT = 0.30

#: Thresholds sampled when tracing the PRO curve. The exact curve would need
#: one threshold per distinct pixel score, which is tens of millions.
_PRO_THRESHOLDS = 200

#: Bad/good-part rate PG2/PB2 are quoted at (Baitieva et al. 2025).
_PRESORT_RATE = 0.02


@dataclass(frozen=True)
class ImageMetrics:
    """Image-level classification quality for one run.

    Attributes:
        auroc: Threshold-free ranking quality.
        average_precision: Area under the precision-recall curve.
        f1_max: Best achievable F1 over all thresholds.
        f1_threshold: The threshold achieving :attr:`f1_max`.
        precision_at_f1: Precision at that threshold.
        recall_at_f1: Recall at that threshold.
        balanced_accuracy: Mean of per-class recall at that threshold.
        error_rate: Misclassification rate at that threshold.
        balanced_error_rate: ``1 - balanced_accuracy``; the honest error rate
            when the classes are as lopsided as Severstal's.
        fpr_at_95tpr: Good parts wrongly rejected when catching 95% of defects.
        fnr_at_1fpr: Defects missed when held to a 1% false-alarm budget.
        pg2: Presorted-Good-at-2% -- the fraction of good parts a fixed-threshold
            sorter correctly passes when that threshold is set to catch all but
            2% of the bad parts. A different operating point than fpr_at_95tpr
            (5% miss budget vs 2%), not a replacement for it.
        pb2: Presorted-Bad-at-2% -- the mirror of pg2: the fraction of bad parts
            correctly rejected when the threshold is set to only false-alarm on
            2% of good parts.
        n_normal: Count of normal test images.
        n_anomalous: Count of anomalous test images.
    """

    auroc: float
    average_precision: float
    f1_max: float
    f1_threshold: float
    precision_at_f1: float
    recall_at_f1: float
    balanced_accuracy: float
    error_rate: float
    balanced_error_rate: float
    fpr_at_95tpr: float
    fnr_at_1fpr: float
    pg2: float
    pb2: float
    n_normal: int
    n_anomalous: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the metrics."""
        return asdict(self)


@dataclass(frozen=True)
class PixelMetrics:
    """Localization quality, where ground-truth masks are available.

    Attributes:
        pixel_auroc: Per-pixel ranking quality.
        pixel_average_precision: Area under the per-pixel PR curve. Far more
            informative than pixel AUROC here, since defective pixels are a
            fraction of a percent of the total.
        aupro: Per-region overlap integrated to 30% FPR, pooled across the
            whole batch's regions and normal pixels into one curve.
        aupimo: Per-*image* analog of aupro -- each image's regions are
            integrated against that image's own normal-pixel FPR range, then
            averaged. Pooling (aupro) lets one image's abundant normal pixels
            dominate the FPR axis for every image's regions; this doesn't.
        n_images: Images that carried a usable mask.
    """

    pixel_auroc: float
    pixel_average_precision: float
    aupro: float
    aupimo: float
    n_images: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the metrics."""
        return asdict(self)


def _empty_image_metrics(n_neg: int, n_pos: int) -> ImageMetrics:
    """Metrics for a split that has only one class, where nothing is defined."""
    nan = float("nan")
    return ImageMetrics(
        auroc=nan,
        average_precision=nan,
        f1_max=nan,
        f1_threshold=nan,
        precision_at_f1=nan,
        recall_at_f1=nan,
        balanced_accuracy=nan,
        error_rate=nan,
        balanced_error_rate=nan,
        fpr_at_95tpr=nan,
        fnr_at_1fpr=nan,
        pg2=nan,
        pb2=nan,
        n_normal=n_neg,
        n_anomalous=n_pos,
    )


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> ImageMetrics:
    """Score image-level predictions against ground truth.

    Args:
        scores: Anomaly scores, higher means more anomalous. Any real-valued
            range is accepted; every metric here is threshold-swept or
            rank-based, so methods with different score scales stay comparable.
        labels: Boolean (or 0/1) ground-truth anomaly labels.

    Returns:
        The populated :class:`ImageMetrics`, with ``nan`` for anything
        undefined because only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    if n_pos == 0 or n_neg == 0:
        return _empty_image_metrics(n_neg, n_pos)

    # A method that fails to produce finite scores must not silently win or
    # lose on NaN ordering; push non-finite scores to the extreme instead.
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float64).max)

    f1, threshold, precision, recall, balanced, error_rate = _best_f1(scores, labels)
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)

    return ImageMetrics(
        auroc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        f1_max=f1,
        f1_threshold=threshold,
        precision_at_f1=precision,
        recall_at_f1=recall,
        balanced_accuracy=balanced,
        error_rate=error_rate,
        balanced_error_rate=1.0 - balanced,
        fpr_at_95tpr=_fpr_at_recall(fpr_curve, tpr_curve, _TARGET_RECALL),
        fnr_at_1fpr=_fnr_at_fpr(fpr_curve, tpr_curve, _FALSE_ALARM_BUDGET),
        pg2=_pg_at_bad_rate(scores, labels, _PRESORT_RATE),
        pb2=_pb_at_good_rate(scores, labels, _PRESORT_RATE),
        n_normal=n_neg,
        n_anomalous=n_pos,
    )


def _fpr_at_recall(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """Smallest false-positive rate that still reaches ``target`` recall."""
    reaching = fpr[tpr >= target]
    return float(reaching.min()) if reaching.size else 1.0


def _fnr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, budget: float) -> float:
    """Miss rate at the best recall achievable within a false-positive budget."""
    affordable = tpr[fpr <= budget]
    return float(1.0 - affordable.max()) if affordable.size else 1.0


def _pg_at_bad_rate(scores: np.ndarray, labels: np.ndarray, bad_rate: float) -> float:
    """Presorted-Good-at-``bad_rate``: fraction of good parts correctly passed.

    Fixes the threshold so exactly ``bad_rate`` of the anomalous parts fall
    below it (i.e. would be missed), then reads off how many normal parts
    also fall below it (i.e. are correctly passed). Structurally the mirror
    of :func:`_fnr_at_fpr`, which fixes the false-positive rate on normals
    and reads the miss rate on anomalies; this fixes the miss rate on
    anomalies and reads the correct-accept rate on normals.
    """
    threshold = np.percentile(scores[labels], bad_rate * 100)
    return float((scores[~labels] < threshold).mean())


def _pb_at_good_rate(scores: np.ndarray, labels: np.ndarray, good_rate: float) -> float:
    """Presorted-Bad-at-``good_rate``: fraction of bad parts correctly rejected.

    Mirror of :func:`_pg_at_bad_rate`: fixes the threshold so exactly
    ``good_rate`` of the normal parts fall at or above it (i.e. would be
    false-alarmed), then reads off how many anomalous parts also clear it
    (i.e. are correctly rejected).
    """
    threshold = np.percentile(scores[~labels], (1.0 - good_rate) * 100)
    return float((scores[labels] >= threshold).mean())


def _best_f1(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Sweep every distinct score as a threshold and keep the best F1.

    Returns:
        ``(f1, threshold, precision, recall, balanced_accuracy, error_rate)``,
        the last two measured at the F1-optimal threshold.
    """
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    # Cumulative counts for "predict anomalous for the top-k scores".
    true_pos = np.cumsum(sorted_labels)
    false_pos = np.cumsum(~sorted_labels)

    # Only thresholds at a score boundary are distinct decision rules.
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_pos = true_pos[distinct]
    false_pos = false_pos[distinct]

    precision = true_pos / np.maximum(true_pos + false_pos, 1)
    recall = true_pos / n_pos
    denominator = np.maximum(precision + recall, 1e-12)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / denominator, 0.0)

    best = int(np.argmax(f1))
    false_neg = n_pos - true_pos[best]
    true_neg = n_neg - false_pos[best]
    balanced = 0.5 * (true_pos[best] / n_pos + true_neg / n_neg)
    error_rate = (false_pos[best] + false_neg) / (n_pos + n_neg)
    return (
        float(f1[best]),
        float(sorted_scores[distinct][best]),
        float(precision[best]),
        float(recall[best]),
        float(balanced),
        float(error_rate),
    )


def compute_pixel_metrics(maps: np.ndarray, masks: np.ndarray) -> PixelMetrics:
    """Score localization quality against ground-truth masks.

    Args:
        maps: ``(N, H, W)`` anomaly maps, higher means more anomalous.
        masks: ``(N, H, W)`` boolean ground-truth defect masks.

    Returns:
        The populated :class:`PixelMetrics`, all ``nan`` if no defective pixel
        is present anywhere in the set.
    """
    maps = np.asarray(maps, dtype=np.float32)
    masks = np.asarray(masks).astype(bool)
    flat_maps = maps.reshape(-1)
    flat_masks = masks.reshape(-1)

    if not flat_masks.any() or flat_masks.all():
        nan = float("nan")
        return PixelMetrics(nan, nan, nan, nan, int(maps.shape[0]))

    flat_maps = np.nan_to_num(flat_maps, nan=0.0)
    return PixelMetrics(
        pixel_auroc=float(roc_auc_score(flat_masks, flat_maps)),
        pixel_average_precision=float(average_precision_score(flat_masks, flat_maps)),
        aupro=compute_aupro(maps, masks),
        aupimo=compute_aupimo(maps, masks)[0],
        n_images=int(maps.shape[0]),
    )


def _integrate_pro_curve(regions: list[np.ndarray], normal_scores: np.ndarray) -> float:
    """Sweep thresholds over ``normal_scores``' range, integrate mean per-region
    overlap against normal-pixel FPR up to :data:`_AUPRO_FPR_LIMIT`.

    Shared by :func:`compute_aupro` (called once, pooled across a whole
    batch) and :func:`compute_aupimo` (called once per image) -- the only
    difference between AUPRO and AUPIMO is *what* ``regions``/
    ``normal_scores`` cover, not how the curve itself is built and integrated.
    """
    if not regions or normal_scores.size == 0:
        return float("nan")

    low = float(min(normal_scores.min(), min(r.min() for r in regions)))
    high = float(max(normal_scores.max(), max(r.max() for r in regions)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float("nan")
    thresholds = np.linspace(high, low, _PRO_THRESHOLDS)

    sorted_normal = np.sort(normal_scores)
    total_normal = sorted_normal.size

    fprs: list[float] = []
    pros: list[float] = []
    for threshold in thresholds:
        exceeding = total_normal - int(np.searchsorted(sorted_normal, threshold, side="left"))
        fpr = exceeding / total_normal
        pro = float(np.mean([float((region >= threshold).mean()) for region in regions]))
        fprs.append(fpr)
        pros.append(pro)
        if fpr > _AUPRO_FPR_LIMIT:
            break

    fpr_array = np.asarray(fprs)
    pro_array = np.asarray(pros)
    keep = fpr_array <= _AUPRO_FPR_LIMIT
    if keep.sum() < 2:
        return float("nan")

    fpr_array = fpr_array[keep]
    pro_array = pro_array[keep]
    order = np.argsort(fpr_array)
    # Normalize by the integration limit so a perfect detector scores 1.0.
    return float(np.trapezoid(pro_array[order], fpr_array[order]) / _AUPRO_FPR_LIMIT)


def compute_aupro(maps: np.ndarray, masks: np.ndarray) -> float:
    """Area under the per-region-overlap curve, integrated to 30% FPR.

    Pixel AUROC is dominated by large defects, because a big region simply
    contributes more pixels. AUPRO instead averages the overlap achieved on
    each connected defect region, so a hairline crack counts exactly as much
    as a large stain -- which is the right weighting when the subtle defects
    are the ones that escape.

    Args:
        maps: ``(N, H, W)`` anomaly maps.
        masks: ``(N, H, W)`` boolean ground-truth masks.

    Returns:
        AUPRO normalized to ``[0, 1]``, or ``nan`` if no regions exist.
    """
    regions: list[np.ndarray] = []
    for image_index in range(masks.shape[0]):
        labeled, count = ndimage.label(masks[image_index])
        for region_id in range(1, count + 1):
            regions.append(maps[image_index][labeled == region_id])

    normal_scores = maps[~masks]
    return _integrate_pro_curve(regions, normal_scores)


def compute_aupimo(maps: np.ndarray, masks: np.ndarray) -> tuple[float, np.ndarray]:
    """Area under the per-IMAGE overlap curve (AUPIMO), best-effort port.

    Vendored per the same reasoning as AUPRO above, following its pattern --
    see Bertoldo et al., "AUPIMO: Redefining PRO for Sparse Anomaly
    Localization" (BMVC 2024). The definitional difference from AUPRO: AUPRO
    pools every region and every normal pixel across the *whole batch* into
    one global FPR axis (so one image's abundant normal pixels can dominate
    the curve every other image's regions get scored against). AUPIMO
    instead integrates each image's own regions against that same image's
    own normal-pixel FPR range, then averages the per-image scores -- no
    image can skew another's curve.

    NOTE: this port has not been cross-checked against the official
    ``jpcbertoldo/aupimo`` reference implementation's numeric output (only
    against hand-constructed bound cases -- see
    ``training/tests/test_evaluation.py``). Treat it as a best-effort
    translation of the definition above, not a verified-bit-exact port.

    Args:
        maps: ``(N, H, W)`` anomaly maps.
        masks: ``(N, H, W)`` boolean ground-truth masks.

    Returns:
        ``(mean_aupimo, per_image_scores)``. ``mean_aupimo`` is ``nan`` if no
        image has both a defect region and normal pixels to score against.
        Images with no anomalous pixels are skipped, same as AUPRO's
        implicit whole-set nan guard for an all-normal set.
    """
    per_image_scores: list[float] = []
    for image_index in range(masks.shape[0]):
        mask = masks[image_index]
        if not mask.any():
            continue
        image_map = maps[image_index]
        labeled, count = ndimage.label(mask)
        regions = [image_map[labeled == region_id] for region_id in range(1, count + 1)]
        normal_scores = image_map[~mask]
        score = _integrate_pro_curve(regions, normal_scores)
        if score == score:  # not NaN
            per_image_scores.append(score)

    if not per_image_scores:
        return float("nan"), np.array([])
    return float(np.mean(per_image_scores)), np.array(per_image_scores)


# -----------------------------------------------------------------------------
# Shared timing helpers (cost and deploy both measure a forward pass the same way)
# -----------------------------------------------------------------------------

#: Forward passes discarded before timing, to let CUDA autotune kernels and
#: allocate its caches. Shared by the cost and deploy stages below.
_WARMUP_RUNS = 10


def _forward_fn(scorer: torch.nn.Module):
    """Return the callable that performs one scoring pass.

    Dinomaly exposes ``model.anomaly_map``; every native ``EmbeddingScorer``
    (PatchCore/PaDiM/DFM) exposes ``embed``/``score_patches``.
    """
    if hasattr(scorer, "model") and hasattr(scorer.model, "anomaly_map"):
        return lambda frame: scorer.model.anomaly_map(frame)
    return lambda frame: scorer.score_patches(scorer.embed(frame))


def _sync(device: torch.device) -> None:
    """Block until queued CUDA work finishes, so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# Cost: deployment-cost instrumentation, measured from a saved checkpoint
# -----------------------------------------------------------------------------
#
# This is the entire reason training.sweep saves a checkpoint for every run:
# a cost pass that loads the fitted model back and times its forward pass,
# instead of re-fitting just to get something to time (which is what the
# deploy stage below did before checkpoints existed, and still does for its
# own separate multi-device CPU/GPU comparison -- this is not a replacement
# for that, it is what backfills inference_latency_ms_p50 and friends onto
# the same SQLite row the accuracy sweep already wrote).
#
# Visits every status="ok" row missing cost columns, loads that run's
# checkpoint, measures latency/throughput/params/VRAM, and writes the
# columns back onto the same row via store.update_columns -- no re-fit, no
# re-score.

#: Timed batch-1 passes for the cost pass. The spec asks for >=100;
#: comfortably clears it.
_COST_TIMED_RUNS = 100


def _time_batch(forward, frame: torch.Tensor, device: torch.device, runs: int) -> list[float]:
    """Warm up, then time ``runs`` forward passes. Returns per-call seconds."""
    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            forward(frame)
        _sync(device)

        durations: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            forward(frame)
            _sync(device)
            durations.append(time.perf_counter() - start)
    return durations


def measure_inference_cost(
    scorer: torch.nn.Module, height: int, width: int, device: str
) -> dict[str, float]:
    """Load-and-time a fitted scorer.

    Args:
        scorer: A checkpoint loaded via ``training.store.load_checkpoint``.
        height: Input height.
        width: Input width.
        device: Torch device string.

    Returns:
        ``inference_latency_ms_p50``, ``inference_latency_ms_p95``,
        ``throughput_fps_bs1``, ``throughput_fps_bs16`` (``nan`` if the
        method's forward path rejects a batch), ``model_params_millions``,
        ``peak_gpu_memory_mb`` (inference-only -- reset immediately before
        this function's own timed loop, distinct from the sweep's
        fit+score-spanning ``peak_vram_gb``).
    """
    torch_device = torch.device(device)
    scorer.to(torch_device).eval()
    if hasattr(scorer, "device"):
        scorer.device = torch_device

    # Some methods (Dinomaly) always run at their own fixed resolution
    # regardless of the dataset -- use that instead of the dataset's declared
    # geometry when the scorer exposes one, or timing would either crash
    # (input not a multiple of the patch size) or measure the wrong shape.
    height, width = getattr(scorer, "input_size", (height, width))

    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    forward = _forward_fn(scorer)

    frame_bs1 = torch.rand(1, 3, height, width, device=torch_device) * 255.0
    durations_bs1 = _time_batch(forward, frame_bs1, torch_device, _COST_TIMED_RUNS)
    latency_ms = sorted(d * 1000.0 for d in durations_bs1)
    p50 = latency_ms[len(latency_ms) // 2]
    p95 = latency_ms[int(len(latency_ms) * 0.95)]
    mean_seconds_bs1 = sum(durations_bs1) / len(durations_bs1)

    try:
        frame_bs16 = torch.rand(16, 3, height, width, device=torch_device) * 255.0
        durations_bs16 = _time_batch(forward, frame_bs16, torch_device, 20)
        mean_seconds_bs16 = sum(durations_bs16) / len(durations_bs16)
        throughput_fps_bs16 = 16.0 / mean_seconds_bs16
    except RuntimeError:
        # The method's forward path doesn't support batching (or this batch
        # doesn't fit in VRAM) -- report it as unknown rather than failing
        # the whole cost measurement.
        throughput_fps_bs16 = float("nan")

    params_millions = sum(p.numel() for p in scorer.parameters()) / 1e6
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch_device.type == "cuda" else 0.0
    )

    return {
        "inference_latency_ms_p50": round(p50, 3),
        "inference_latency_ms_p95": round(p95, 3),
        "throughput_fps_bs1": round(1.0 / mean_seconds_bs1, 2),
        "throughput_fps_bs16": round(throughput_fps_bs16, 2)
        if throughput_fps_bs16 == throughput_fps_bs16  # not NaN
        else float("nan"),
        "model_params_millions": round(params_millions, 3),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 1),
    }


def _checkpoint_config_key(row) -> str:
    """The config key a row's checkpoint was saved under.

    Multi-class checkpoints are saved once per family (``row.dataset``, e.g.
    ``"mvtec"``), not per category -- every category's row shares that one
    fit. One-class and few-shot checkpoints are saved per the row's own
    ``config`` (few-shot's shot-specific ``regime`` string already
    disambiguates the path, same as at save time).
    """
    return row.dataset if row.regime == "multiclass" else row.config


def run_cost_pass(results_db: Path, checkpoint_root: Path, device: str) -> tuple[int, int]:
    """Visit every completed row missing cost data and backfill it.

    Returns:
        ``(updated, skipped_no_checkpoint)`` counts.
    """
    from sqlalchemy import select as sa_select

    from training.store import RunRow, session_scope

    _, session_factory = store.open_database(results_db)

    with session_scope(session_factory) as session:
        rows = list(
            session.scalars(
                sa_select(RunRow).where(
                    RunRow.status == "ok", RunRow.inference_latency_ms_p50.is_(None)
                )
            )
        )
        # Copy out the plain values we need -- the ORM objects go stale once
        # this session closes, and update_columns opens its own session per row.
        pending = [
            (row.run_id, row.regime, row.method, _checkpoint_config_key(row), row.height, row.width)
            for row in rows
        ]

    updated = 0
    skipped = 0
    for index, (run_id, regime, method, config_key, height, width) in enumerate(pending, start=1):
        path = store.checkpoint_path(checkpoint_root, regime, method, config_key)
        scorer = store.load_checkpoint(path, device=device)
        if scorer is None:
            print(
                f"[{index}/{len(pending)}] {method} @ {config_key}: "
                f"no checkpoint at {path}, skipping"
            )
            skipped += 1
            continue
        cost = measure_inference_cost(scorer, height, width, device)
        store.update_columns(session_factory, run_id, cost)
        print(f"[{index}/{len(pending)}] {method} @ {config_key}: {cost}")
        updated += 1
        del scorer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return updated, skipped


def main_cost(argv: list[str] | None = None) -> None:
    """Parse arguments and run the cost pass."""
    parser = argparse.ArgumentParser(description=main_cost.__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    updated, skipped = run_cost_pass(args.results_db, args.checkpoints, args.device)
    print(f"\nbackfilled cost metrics on {updated} row(s), {skipped} skipped (no checkpoint found)")


# -----------------------------------------------------------------------------
# Deploy: multi-device deployment profiling for a short list of top methods
# -----------------------------------------------------------------------------
#
# The lab card is not the production card. A benchmark that reports accuracy
# alone will happily recommend a ViT-L memory bank for a station with a 40 ms
# cycle time, and the recommendation will be wrong. This measures the numbers
# that decide deployability: GPU latency (p50/p95) at batch 1 (how a station
# actually runs), CPU latency (many edge boxes have no GPU at all), model
# size and peak VRAM (what fits on the box), and throughput in parts per
# minute (the number a plant manager actually asks for). p95 rather than
# mean: a station that misses its cycle time one frame in twenty has a
# defect-escape problem, and a mean hides that entirely.
#
# Unlike the cost stage above (which loads an *already-fitted* checkpoint
# from the sweep), this stage fits each method itself, from scratch, so it
# can profile a device the sweep never ran on (typically CPU, alongside the
# sweep's own GPU).

#: Timed passes for the deploy stage. Enough for a stable p95 without
#: dominating the sweep.
_DEPLOY_TIMED_RUNS = 60


def _percentile_latency(durations: list[float]) -> dict[str, float]:
    """Summarize per-call durations (seconds) as latency and throughput."""
    array = np.asarray(durations) * 1000.0
    p50 = float(np.percentile(array, 50))
    p95 = float(np.percentile(array, 95))
    return {
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "latency_mean_ms": round(float(array.mean()), 3),
        # Parts per minute at the p95 latency, i.e. the rate the line can be
        # guaranteed rather than the rate it averages.
        "throughput_ppm_p95": round(60_000.0 / max(p95, 1e-6), 1),
    }


def measure_latency(
    scorer: torch.nn.Module, height: int, width: int, device: str
) -> dict[str, float]:
    """Time single-frame inference on one device.

    Args:
        scorer: A fitted scorer exposing ``embed`` and ``patch_scores``, or a
            model exposing ``anomaly_map``.
        height: Input height.
        width: Input width.
        device: Torch device string.

    Returns:
        Latency and throughput statistics.
    """
    torch_device = torch.device(device)
    scorer.to(torch_device).eval()
    if hasattr(scorer, "device"):
        scorer.device = torch_device

    frame = torch.rand(1, 3, height, width, device=torch_device) * 255.0
    forward = _forward_fn(scorer)

    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            forward(frame)
        _sync(torch_device)

        durations: list[float] = []
        for _ in range(_DEPLOY_TIMED_RUNS):
            start = time.perf_counter()
            forward(frame)
            _sync(torch_device)
            durations.append(time.perf_counter() - start)

    return _percentile_latency(durations)


def _parameter_bytes(scorer: torch.nn.Module) -> int:
    """Total bytes held by parameters and buffers.

    Buffers matter more than parameters here: a PatchCore memory bank is
    entirely buffer, and it is what makes the artifact large.
    """
    total = sum(p.numel() * p.element_size() for p in scorer.parameters())
    total += sum(b.numel() * b.element_size() for b in scorer.buffers() if b is not None)
    return total


def profile_method(
    method_name: str,
    config_key: str,
    data_root: Path,
    options: RunOptions,
    devices: tuple[str, ...],
) -> dict[str, Any]:
    """Fit one method and profile its deployment characteristics.

    Returns:
        A profile row; ``status`` is ``"ok"`` or ``"failed"``.
    """
    from training.data import load_split, parse_config

    row: dict[str, Any] = {"method": method_name, "config": config_key}
    try:
        spec = get(method_name)
        config = parse_config(config_key, data_root)
        train_paths, test_split = load_split(config, data_root)
        train_paths = subsample_fit(train_paths, options.max_fit_images, options.seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        scorer = spec.fit(config, train_paths, test_split, options)

        row.update(
            status="ok",
            exportable=spec.exportable,
            family=spec.family,
            height=config.height,
            width=config.width,
            model_mb=round(_parameter_bytes(scorer) / 1e6, 2),
            fit_peak_vram_gb=round(
                torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0, 3
            ),
        )

        for device in devices:
            if device.startswith("cuda") and not torch.cuda.is_available():
                continue
            stats = measure_latency(scorer, config.height, config.width, device)
            row.update({f"{device}_{key}": value for key, value in stats.items()})

    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _leaderboard_methods(ranking_csv: Path, limit: int) -> list[str]:
    """Read the top-ranked method names from the leaderboard's ranking table.

    Raises:
        SystemExit: If the ranking has not been generated yet.
    """
    if not ranking_csv.exists():
        msg = f"No ranking at {ranking_csv}. Run `python -m training.evaluate leaderboard` first."
        raise SystemExit(msg)
    frame = pd.read_csv(ranking_csv)
    return [str(name) for name in frame["method"].head(limit)]


def main_deploy(argv: list[str] | None = None) -> None:
    """Parse arguments and profile the selected methods.

    Usage:
        python -m training.evaluate deploy --methods patchcore_wide_resnet50_2 dinomaly_vitb14
        python -m training.evaluate deploy --from-leaderboard --dataset mvtec/bottle
    """
    parser = argparse.ArgumentParser(description=main_deploy.__doc__)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--from-leaderboard", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--ranking", type=Path, default=RESULTS_DIR / "ranking.csv")
    parser.add_argument("--dataset", default="mvtec/bottle", help="Config used for the fit.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--max-fit-images", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "deployment.jsonl")
    args = parser.parse_args(argv)

    if args.from_leaderboard:
        methods = _leaderboard_methods(args.ranking, args.top)
    elif args.methods:
        methods = args.methods
    else:
        msg = "Pass --methods, or --from-leaderboard."
        raise SystemExit(msg)

    options = RunOptions(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_fit_images=args.max_fit_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        epochs=args.epochs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"profiling {len(methods)} methods on {args.dataset}\n")
    print(f"{'method':<38s} {'MB':>8s} {'gpu p95':>9s} {'cpu p95':>9s} {'ppm':>8s}")

    with args.output.open("w", encoding="utf-8") as handle:
        for method in methods:
            row = profile_method(method, args.dataset, args.data_root, options, tuple(args.devices))
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            if row["status"] == "ok":
                print(
                    f"{method:<38.38s} {row.get('model_mb', 0):>8.1f} "
                    f"{row.get('cuda_latency_p95_ms', float('nan')):>9.1f} "
                    f"{row.get('cpu_latency_p95_ms', float('nan')):>9.1f} "
                    f"{row.get('cuda_throughput_ppm_p95', float('nan')):>8.0f}"
                )
            else:
                print(f"{method:<38.38s} FAILED {row['error']}")

    print(f"\nWrote {args.output}")


# -----------------------------------------------------------------------------
# Metrics backfill: PG2/PB2/AUPIMO onto existing rows, no rerun required
# -----------------------------------------------------------------------------
#
# Both new metrics can be computed from what the sweep already saved: PG2/PB2
# (image-level) need only the raw scores/labels already in the .npz
# artifact; AUPIMO (pixel-level) needs the artifact's saved anomaly maps
# paired with freshly-loaded ground-truth masks -- the .npz doesn't store
# masks (only scores/labels/paths/maps), so this reloads them from disk the
# same way the sweep did (cheap: no model, no GPU, just image I/O), using the
# same class-balanced sampling as the sweep's own pixel-metrics pass so the
# backfilled number reflects the same sample the original aupro was scored
# against.


def _pixel_backfill(
    artifact, config: DatasetConfig, data_root: Path, seed: int
) -> dict[str, float] | None:
    """Recompute pixel metrics (for aupimo) over the same sample the sweep used."""
    if artifact.maps is None or not has_masks(config):
        return None
    test_split = [
        (Path(p), bool(label)) for p, label in zip(artifact.paths, artifact.labels, strict=True)
    ]
    indices = pixel_sample_indices(test_split, seed)

    selected_maps = []
    selected_masks = []
    for index in indices:
        mask = load_mask(test_split[index][0], config, data_root, config.height, config.width)
        if mask is None:
            continue
        selected_maps.append(artifact.maps[index])
        selected_masks.append(mask)

    if not selected_masks:
        return None

    pixel = compute_pixel_metrics(np.stack(selected_maps), np.stack(selected_masks))
    return {"aupimo": pixel.aupimo}


def run_metrics_pass(results_db: Path, artifact_root: Path, data_root: Path) -> tuple[int, int]:
    """Visit every completed row missing pg2 and backfill pg2/pb2 (+ aupimo where masks exist).

    Returns:
        ``(updated, skipped_no_artifact)`` counts.
    """
    from sqlalchemy import select as sa_select

    from training.store import RunRow, session_scope

    _, session_factory = store.open_database(results_db)

    with session_scope(session_factory) as session:
        rows = list(
            session.scalars(sa_select(RunRow).where(RunRow.status == "ok", RunRow.pg2.is_(None)))
        )
        pending = [
            (
                row.run_id,
                row.regime,
                row.method,
                row.config,
                row.dataset,
                row.category,
                row.height,
                row.width,
                row.seed,
            )
            for row in rows
        ]

    updated = 0
    skipped = 0
    for index, row in enumerate(pending, start=1):
        run_id, regime, method, config_key, dataset, category, height, width, seed = row
        path = store.artifact_path(artifact_root, regime, method, config_key, seed)
        artifact = store.load_artifact(path)
        if artifact is None:
            print(
                f"[{index}/{len(pending)}] {method} @ {config_key}: no artifact at {path}, skipping"
            )
            skipped += 1
            continue

        columns: dict[str, float] = {}
        if artifact.labels.any() and not artifact.labels.all():
            image_metrics = compute_metrics(artifact.scores, artifact.labels)
            columns["pg2"] = image_metrics.pg2
            columns["pb2"] = image_metrics.pb2

        config = DatasetConfig(
            dataset=dataset, category=category, height=height, width=width, position_aligned=True
        )
        pixel_columns = _pixel_backfill(artifact, config, data_root, seed)
        if pixel_columns:
            columns.update(pixel_columns)

        if columns:
            store.update_columns(session_factory, run_id, columns)
            print(f"[{index}/{len(pending)}] {method} @ {config_key}: {columns}")
            updated += 1
        else:
            skipped += 1

    return updated, skipped


def main_metrics_backfill(argv: list[str] | None = None) -> None:
    """Parse arguments and run the metrics backfill pass."""
    parser = argparse.ArgumentParser(description=main_metrics_backfill.__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    args = parser.parse_args(argv)

    updated, skipped = run_metrics_pass(args.results_db, args.artifacts, args.data_root)
    print(f"\nbackfilled pg2/pb2(/aupimo) on {updated} row(s), {skipped} skipped")


# -----------------------------------------------------------------------------
# Ensemble: score fusion across methods, computed from stored artifacts
# -----------------------------------------------------------------------------
#
# Different families fail differently. A memory bank misses a defect whose
# patches happen to resemble something in the normal set; a reconstruction
# model misses one it has learned to redraw; a Gaussian misses one that stays
# inside the normal ellipsoid. Where those failures are uncorrelated,
# combining the scores recovers them -- which is why an ensemble is worth
# measuring even when no member is best on its own.
#
# This runs entirely on the .npz artifacts the sweep already wrote, so
# evaluating every pair and triple costs seconds and no GPU. Fusion happens
# on ranks, not raw scores: members produce Mahalanobis distances,
# nearest-neighbour distances and cosine errors, whose scales differ by
# orders of magnitude -- averaging those directly would just return
# whichever member has the largest numbers.

#: Artifact filenames look like ``{method}__{config_slug}__seed{N}.npz``.
_ARTIFACT_NAME = re.compile(r"^(?P<method>.+)__(?P<slug>.+)__seed(?P<seed>\d+)$")

#: Candidate members are capped to the best few per family. Fusing two
#: backbones of the same method mostly averages correlated errors; the gain
#: comes from combining *different* families.
_PER_FAMILY_CANDIDATES = 2


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map scores to ``[0, 1]`` by rank, so members combine on equal footing.

    Ties receive their average rank, which keeps a member that emits many
    identical scores from silently dominating the fusion.
    """
    from scipy.stats import rankdata

    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    return (rankdata(scores) - 1) / (scores.size - 1)


def fuse(score_sets: list[np.ndarray], rule: str) -> np.ndarray:
    """Combine rank-normalized member scores under one fusion rule.

    Args:
        score_sets: One score array per member, all the same length.
        rule: ``"mean"``, ``"max"``, or ``"gmean"``.

    Returns:
        The fused score array.

    Raises:
        ValueError: If ``rule`` is unknown.
    """
    stacked = np.stack([rank_normalize(scores) for scores in score_sets])
    if rule == "mean":
        return stacked.mean(axis=0)
    if rule == "max":
        # Any member confidently flagging an image carries the decision.
        # Best recall, worst false-alarm rate.
        return stacked.max(axis=0)
    if rule == "gmean":
        # Geometric mean: members must broadly agree, so it suppresses the
        # lone-member false alarms that "max" lets through.
        return np.exp(np.log(np.clip(stacked, 1e-9, None)).mean(axis=0))
    msg = f"Unknown fusion rule: {rule!r}"
    raise ValueError(msg)


def discover_artifacts(artifact_root: Path, regime: str) -> dict[str, dict[str, Path]]:
    """Index stored artifacts as ``{config_key: {method: path}}``.

    Fusion stays single-seed: with the 3-seed repeats each producing its own
    ``__seed{N}.npz``, this keeps only the lowest seed per (config, method)
    rather than combinatorially fusing across seeds too, which would blow up
    the already-expensive combination search for a question ("does seed
    variance affect ensembling") this pass isn't trying to answer.
    """
    index: dict[str, dict[str, Path]] = defaultdict(dict)
    chosen_seed: dict[tuple[str, str], int] = {}
    regime_dir = artifact_root / regime
    if not regime_dir.is_dir():
        return index
    for path in sorted(regime_dir.glob("*.npz")):
        match = _ARTIFACT_NAME.match(path.stem)
        if match is None:
            continue
        method, slug, seed = match["method"], match["slug"], int(match["seed"])
        key = (slug, method)
        if key not in chosen_seed or seed < chosen_seed[key]:
            chosen_seed[key] = seed
            index[slug][method] = path
    return index


def choose_candidates(
    results_db: Path, regime: str, per_family: int = _PER_FAMILY_CANDIDATES
) -> list[str]:
    """Pick the strongest few methods per family as ensemble candidates.

    Args:
        results_db: The sweep's SQLite store.
        regime: Regime to read scores from.
        per_family: How many methods to keep from each family.

    Returns:
        Candidate method names.
    """
    from sqlalchemy import select as sa_select

    from training.store import RunRow, session_scope

    _, session_factory = store.open_database(results_db)
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    with session_scope(session_factory) as session:
        rows = session.scalars(
            sa_select(RunRow).where(RunRow.status == "ok", RunRow.regime == regime)
        )
        for row in rows:
            if row.auroc is None or row.auroc != row.auroc:  # missing or NaN
                continue
            scores[(row.family, row.method)].append(float(row.auroc))

    best: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (family, method), values in scores.items():
        best[family].append((float(np.mean(values)), method))

    candidates: list[str] = []
    for entries in best.values():
        entries.sort(reverse=True)
        candidates.extend(method for _, method in entries[:per_family])
    return sorted(candidates)


def evaluate_combinations(
    artifact_root: Path,
    regime: str,
    candidates: list[str],
    max_members: int,
    rules: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Score every candidate combination on every config it fully covers.

    A combination is only evaluated where *every* member has an artifact with
    matching labels; a partially-covered ensemble would be scored on an easier
    subset than its members were.

    Returns:
        One row per ``(combination, rule, config)``.
    """
    index = discover_artifacts(artifact_root, regime)
    rows: list[dict[str, Any]] = []

    combos: list[tuple[str, ...]] = []
    for size in range(2, max_members + 1):
        combos.extend(itertools.combinations(candidates, size))

    for slug, by_method in sorted(index.items()):
        # The membership guard has to precede the lookup: comprehension
        # conditions evaluate in written order, so testing it second would
        # raise KeyError on any candidate this config never ran.
        loaded = {
            method: artifact
            for method in candidates
            if method in by_method
            if (artifact := store.load_artifact(by_method[method])) is not None
        }
        if len(loaded) < 2:
            continue

        for combo in combos:
            if not all(method in loaded for method in combo):
                continue
            members = [loaded[method] for method in combo]
            labels = members[0].labels
            if any(
                m.labels.shape != labels.shape or not np.array_equal(m.labels, labels)
                for m in members
            ):
                continue

            for rule in rules:
                fused = fuse([m.scores for m in members], rule)
                metrics = compute_metrics(fused, labels)
                rows.append(
                    {
                        "regime": regime,
                        "method": f"ensemble[{rule}]:" + "+".join(combo),
                        "family": "ensemble",
                        "backend": "ensemble",
                        "config": slug.replace("_", "/", 1) if "_" in slug else slug,
                        "members": list(combo),
                        "rule": rule,
                        "n_members": len(combo),
                        "status": "ok",
                        **metrics.as_dict(),
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-config ensemble rows into a per-combination ranking."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary = [
        {
            "method": method,
            "rule": entries[0]["rule"],
            "n_members": entries[0]["n_members"],
            "members": entries[0]["members"],
            "configs": len(entries),
            "mean_auroc": float(np.mean([e["auroc"] for e in entries])),
            "mean_ap": float(np.mean([e["average_precision"] for e in entries])),
            "mean_f1": float(np.mean([e["f1_max"] for e in entries])),
            "mean_scrap_at_95": float(np.mean([e["fpr_at_95tpr"] for e in entries])),
        }
        for method, entries in grouped.items()
    ]
    summary.sort(key=lambda row: (-row["configs"], -row["mean_auroc"]))
    return summary


def main_ensemble(argv: list[str] | None = None) -> None:
    """Parse arguments, evaluate ensembles, and write the results.

    Usage:
        python -m training.evaluate ensemble
        python -m training.evaluate ensemble --regime multiclass --max-members 3
    """
    parser = argparse.ArgumentParser(description=main_ensemble.__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--regime", default="oneclass")
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--rules", nargs="+", default=["mean", "max", "gmean"])
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ensembles.jsonl")
    args = parser.parse_args(argv)

    if not args.results_db.exists():
        msg = f"No results at {args.results_db}. Run `python -m training.sweep` first."
        raise SystemExit(msg)

    candidates = choose_candidates(args.results_db, args.regime)
    if len(candidates) < 2:
        msg = f"Need at least two successful methods in regime {args.regime!r} to ensemble."
        raise SystemExit(msg)

    print(f"candidates ({len(candidates)}): {', '.join(candidates)}")
    rows = evaluate_combinations(
        args.artifacts, args.regime, candidates, args.max_members, tuple(args.rules)
    )
    if not rows:
        msg = "No artifacts covered a full combination. Was the sweep run with artifacts enabled?"
        raise SystemExit(msg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")

    summary = summarize(rows)
    print(f"\n{len(summary)} combinations over {len({r['config'] for r in rows})} configs\n")
    print(f"{'combination':<64s} {'cfgs':>5s} {'AUROC':>8s} {'F1':>8s} {'scrap@95':>9s}")
    for row in summary[:20]:
        print(
            f"{row['method']:<64.64s} {row['configs']:>5d} "
            f"{row['mean_auroc']:>8.4f} {row['mean_f1']:>8.4f} {row['mean_scrap_at_95']:>9.4f}"
        )
    print(f"\nWrote {args.output}")


# -----------------------------------------------------------------------------
# Leaderboard: aggregate sweep results into a ranking and a written report
# -----------------------------------------------------------------------------
#
# A pile of per-run AUROC numbers is not an answer to "which model should we
# ship". This turns the SQLite store into the things that are: a ranking per
# regime, restricted to methods that actually completed every configuration;
# the regime comparison, which is the real finding (whether one model can
# cover every category or whether the project ships one per category);
# per-dataset winners; and operating-point error rates plus a latency column.
# Every metric is reported as mean +/- std across seeds, never a bare number
# -- see aggregate_seeds.

#: Columns aggregated as a mean, with the label used in the report. Cost
#: columns are aggregated the same way as accuracy ones -- the whole point of
#: capturing them per-run is that they get the same mean+/-std treatment.
_MEAN_COLUMNS = {
    "auroc": "mean_auroc",
    "average_precision": "mean_ap",
    "f1_max": "mean_f1",
    "fpr_at_95tpr": "scrap_at_95",
    "fnr_at_1fpr": "escape_at_1fpr",
    "pg2": "mean_pg2",
    "pb2": "mean_pb2",
    "balanced_error_rate": "bal_error",
    "aupro": "mean_aupro",
    "aupimo": "mean_aupimo",
    "pixel_auroc": "mean_pixel_auroc",
    "ms_per_image": "ms_per_image",
    "fit_seconds": "fit_seconds",
    "inference_latency_ms_p50": "latency_p50_ms",
    "inference_latency_ms_p95": "latency_p95_ms",
    "throughput_fps_bs1": "throughput_fps_bs1",
    "throughput_fps_bs16": "throughput_fps_bs16",
    "model_params_millions": "model_params_m",
    "peak_gpu_memory_mb": "peak_gpu_mb",
    "training_wall_clock_seconds": "train_wall_clock_s",
}

#: Metrics that are prevalence-sensitive and therefore unsafe to compare
#: across datasets with different positive rates (Severstal's ~92% positive
#: test split vs the others' much lower rates). AUROC and PG2/PB2 are kept --
#: see the README's Severstal caveat.
_PREVALENCE_SENSITIVE_COLUMNS = ("mean_ap", "mean_f1")


def load_results(results_db: Path) -> pd.DataFrame:
    """Read the sweep's SQLite store into a frame.

    Raises:
        SystemExit: If the database is missing or holds no rows.
    """
    if not results_db.exists():
        msg = f"No results at {results_db}. Run `python -m training.sweep` first."
        raise SystemExit(msg)

    engine, _ = store.open_readonly(results_db)
    frame = pd.read_sql_table("runs", engine)
    if frame.empty:
        msg = f"{results_db} has no rows."
        raise SystemExit(msg)

    for column in (*_MEAN_COLUMNS, "peak_vram_gb"):
        if column not in frame:
            frame[column] = float("nan")
    return frame


def aggregate_seeds(
    frame: pd.DataFrame,
    group_cols: tuple[str, ...] = ("method", "family", "backend", "config", "regime"),
) -> pd.DataFrame:
    """Collapse the seed dimension: every metric becomes ``{col}_mean``/``{col}_std``.

    ``std`` is ``NaN`` (not 0) for a cell with a single seed -- pandas' native
    ``ddof=1`` behavior needs no special-casing here, and a ``NaN`` std
    correctly reads as "no seed variability measured" rather than "measured
    and found to be zero".
    """
    metric_columns = list(dict.fromkeys([*_MEAN_COLUMNS, "peak_vram_gb"]))
    metric_columns = [c for c in metric_columns if c in frame.columns]
    grouped = frame.groupby(list(group_cols))
    agg = grouped[metric_columns].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_seeds"] = grouped.size()
    return agg.reset_index()


def rank_regime(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Build the per-method ranking for one regime.

    Two-level aggregation, per the spec: raw per-seed rows first collapse to
    one seed-mean per ``(method, config)`` cell (:func:`aggregate_seeds`),
    then this groupby averages *those cell means* across configs -- so a
    method that happened to get more successful seeds on one easy config
    cannot silently outweigh the rest.
    """
    subset = frame[(frame["status"] == "ok") & (frame["regime"] == regime)]
    if subset.empty:
        return pd.DataFrame()

    total_configs = frame[frame["regime"] == regime]["config"].nunique()

    cell_means = aggregate_seeds(subset, group_cols=("method", "family", "backend", "config"))

    aggregations: dict[str, Any] = {
        "configs": ("config", "nunique"),
        "min_auroc": ("auroc_mean", "min"),
        "peak_vram_gb": ("peak_vram_gb_mean", "max"),
    }
    aggregations.update(
        {label: (f"{column}_mean", "mean") for column, label in _MEAN_COLUMNS.items()}
    )
    # The "typical" std shown per method is the mean of its per-cell stds --
    # a summary of observed seed variability, not a properly pooled variance
    # (which would need per-cell sample sizes and isn't worth the complexity
    # for a leaderboard reading).
    aggregations.update(
        {f"{label}_std": (f"{column}_std", "mean") for column, label in _MEAN_COLUMNS.items()}
    )

    ranking = cell_means.groupby(["method", "family", "backend"]).agg(**aggregations).reset_index()
    ranking["complete"] = ranking["configs"] == total_configs
    # Complete runs rank above partial ones regardless of their mean, so a
    # method that only survived the easy categories cannot take the top slot.
    ranking = ranking.sort_values(["complete", "mean_auroc"], ascending=[False, False]).reset_index(
        drop=True
    )
    ranking.insert(0, "rank", ranking.index + 1)
    ranking.insert(1, "regime", regime)

    # Severstal's held-out "normal" pool is synthesized (see training.data)
    # and its test split runs ~92% positive vs the other corpora's much lower
    # rate -- AP/F1 are prevalence-sensitive, so blending a Severstal row into
    # a multi-dataset mean_ap/mean_f1 would silently favor whichever dataset
    # mix a method happened to run on. AUROC and PG2/PB2 stay: less
    # prevalence-sensitive, and the whole reason those two exist. This only
    # fires when Severstal is actually mixed with other datasets in this
    # view -- a Severstal-only sweep keeps its AP/F1.
    datasets_in_view = set(subset["dataset"])
    if "severstal" in datasets_in_view and len(datasets_in_view) > 1:
        drop_columns = [
            column
            for label in _PREVALENCE_SENSITIVE_COLUMNS
            for column in (label, f"{label}_std")
            if column in ranking
        ]
        ranking = ranking.drop(columns=drop_columns)
        ranking.attrs["severstal_ap_f1_hidden"] = True
    return ranking


def shared_configs(frame: pd.DataFrame, regimes: list[str]) -> set[str]:
    """Configurations that every named regime actually covers.

    Multi-class is only defined for a family with more than one category, so
    it never covers the single-category corpora that one-class does. Comparing
    a 27-config mean against a 29-config mean would attribute the difference
    to the regime when part of it is just a different set of categories.
    """
    ok = frame[frame["status"] == "ok"]
    per_regime = [set(ok[ok["regime"] == regime]["config"]) for regime in regimes]
    per_regime = [configs for configs in per_regime if configs]
    if not per_regime:
        return set()
    return set.intersection(*per_regime)


def regime_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean AUROC per method per regime, over the configurations they share.

    This is the table the whole study exists to produce: a method whose
    multi-class column matches its one-class column is a method that ships as
    one checkpoint instead of one per category. It is restricted to shared
    configurations so the columns differ by regime alone. Seed-averaged first
    (one value per (method, config, regime) cell) so a multi-seed method isn't
    weighted by how many seeds happened to succeed.
    """
    ok = frame[frame["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()

    regimes = sorted(set(ok["regime"]))
    common = shared_configs(frame, regimes)
    if not common:
        return pd.DataFrame()

    subset = ok[ok["config"].isin(common)]
    cell_means = aggregate_seeds(subset, group_cols=("method", "config", "regime"))
    table = (
        cell_means.pivot_table(
            index="method", columns="regime", values="auroc_mean", aggfunc="mean"
        )
        .round(4)
        .reset_index()
    )
    sort_column = "multiclass" if "multiclass" in table.columns else table.columns[-1]
    return table.sort_values(by=sort_column, ascending=False)


def per_dataset(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Mean AUROC per method per dataset family, within one regime.

    Seed-averaged first, same reasoning as :func:`regime_comparison`.
    """
    subset = frame[(frame["status"] == "ok") & (frame["regime"] == regime)]
    if subset.empty:
        return pd.DataFrame()
    cell_means = aggregate_seeds(subset, group_cols=("method", "dataset", "regime"))
    return (
        cell_means.pivot_table(
            index="method", columns="dataset", values="auroc_mean", aggfunc="mean"
        )
        .round(4)
        .reset_index()
    )


def winners(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Best method per configuration, within one regime.

    Ranked by each (method, config) cell's seed-averaged AUROC, so a single
    lucky seed can't make a mediocre method look like the winner.
    """
    subset = frame[(frame["status"] == "ok") & (frame["regime"] == regime)]
    if subset.empty:
        return pd.DataFrame()
    cell_means = aggregate_seeds(subset, group_cols=("method", "config", "regime"))
    best = cell_means.groupby("config")["auroc_mean"].idxmax()
    columns = [
        "config",
        "method",
        "auroc_mean",
        "average_precision_mean",
        "f1_max_mean",
        "ms_per_image_mean",
    ]
    picked = cell_means.loc[best, [c for c in columns if c in cell_means]].sort_values("config")
    return picked.rename(columns=lambda c: c.removesuffix("_mean"))


def _round(frame: pd.DataFrame, digits: dict[str, int]) -> pd.DataFrame:
    """Round selected columns for display, leaving missing ones alone."""
    display = frame.copy()
    for column, places in digits.items():
        if column in display:
            display[column] = display[column].round(places)
    return display


_DISPLAY_DIGITS = {
    "mean_auroc": 4,
    "min_auroc": 4,
    "mean_ap": 4,
    "mean_f1": 4,
    "scrap_at_95": 4,
    "escape_at_1fpr": 4,
    "mean_pg2": 4,
    "mean_pb2": 4,
    "bal_error": 4,
    "mean_aupro": 4,
    "mean_aupimo": 4,
    "mean_pixel_auroc": 4,
    "ms_per_image": 1,
    "fit_seconds": 1,
    "peak_vram_gb": 2,
    "latency_p50_ms": 1,
    "latency_p95_ms": 1,
    "throughput_fps_bs1": 1,
    "throughput_fps_bs16": 1,
    "model_params_m": 2,
    "peak_gpu_mb": 1,
    "train_wall_clock_s": 1,
}


def _with_mean_std_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace every ``{label}``/``{label}_std`` pair with one "mean +/- std" string.

    Display-only: the numeric ``_mean``/``_std`` columns stay untouched in
    whatever frame gets written to CSV -- this only ever runs on a copy meant
    for the Markdown report, per the spec's "mean +/- std, never a bare
    number" requirement.
    """
    display = frame.copy()
    for label in _MEAN_COLUMNS.values():
        std_label = f"{label}_std"
        if label not in display or std_label not in display:
            continue
        digits = _DISPLAY_DIGITS.get(label, 4)
        display[label] = [
            f"{mean:.{digits}f} +/- {std:.{digits}f}" if pd.notna(std) else f"{mean:.{digits}f}"
            for mean, std in zip(display[label], display[std_label], strict=True)
        ]
        display = display.drop(columns=[std_label])
    return display


def format_report(frame: pd.DataFrame, rankings: dict[str, pd.DataFrame]) -> str:
    """Render the summary tables as a Markdown report."""
    ok = frame[frame["status"] == "ok"]
    failed = frame[frame["status"] != "ok"]

    lines: list[str] = ["# Anomaly-detection model benchmark\n"]
    lines.append(
        f"{ok['method'].nunique()} methods across "
        f"{frame['config'].nunique()} dataset configurations and "
        f"{frame['regime'].nunique()} regimes -- "
        f"{len(ok)} successful runs, {len(failed)} failed.\n"
    )

    multiclass = rankings.get("multiclass", pd.DataFrame())
    oneclass = rankings.get("oneclass", pd.DataFrame())

    if not multiclass.empty:
        complete = multiclass[multiclass["complete"]]
        best = (complete if not complete.empty else multiclass).iloc[0]
        lines.append(
            f"**Headline -- one model, every category.** `{best['method']}` reaches "
            f"**{best['mean_auroc']:.4f} mean AUROC** across all {int(best['configs'])} "
            f"configurations from a single fitted model per dataset family "
            f"(worst config {best['min_auroc']:.4f}, {best['ms_per_image']:.1f} ms/image).\n"
        )
        if not oneclass.empty:
            # Compare on shared configurations only, or the gap conflates the
            # regime with the different set of categories each one covers.
            # Seed-averaged per cell first, same reasoning as rank_regime.
            common = shared_configs(frame, ["multiclass", "oneclass"])
            ok = frame[(frame["status"] == "ok") & (frame["config"].isin(common))]
            cell_means = aggregate_seeds(ok, group_cols=("method", "config", "regime"))
            multi_mean = cell_means[cell_means["regime"] == "multiclass"].groupby("method")[
                "auroc_mean"
            ].mean()
            one_mean = cell_means[cell_means["regime"] == "oneclass"].groupby("method")[
                "auroc_mean"
            ].mean()

            if not multi_mean.empty and not one_mean.empty:
                multi_best = multi_mean.idxmax()
                one_best = one_mean.idxmax()
                gap = multi_mean.max() - one_mean.max()
                lines.append(
                    f"On the {len(common)} configurations both regimes cover, the best "
                    f"multi-class model (`{multi_best}`, {multi_mean.max():.4f}) lands "
                    f"{gap * 100:+.2f} points against the best per-category model "
                    f"(`{one_best}`, {one_mean.max():.4f}) -- which needs one checkpoint "
                    f"per category rather than one per dataset family.\n"
                )

    comparison = regime_comparison(frame)
    if not comparison.empty:
        common = shared_configs(frame, sorted(set(frame[frame["status"] == "ok"]["regime"])))
        lines.append("## Regime comparison\n")
        lines.append(
            f"Mean AUROC per method under each deployment regime, over the "
            f"{len(common)} configurations every regime covers. A method whose "
            "multi-class column matches its one-class column ships as one artifact "
            "per dataset family instead of one per category.\n"
        )
        lines.append(comparison.to_markdown(index=False))
        lines.append("")

    for regime, ranking in rankings.items():
        if ranking.empty:
            continue
        lines.append(f"## Ranking -- {regime}\n")
        lines.append(
            "`scrap_at_95` is the share of good parts rejected when tuned to catch 95% of "
            "defects; `escape_at_1fpr` is the share of defects missed within a 1% "
            "false-alarm budget. Methods that did not complete every configuration are "
            "listed last and are not comparable to a complete row.\n"
        )
        if ranking.attrs.get("severstal_ap_f1_hidden"):
            lines.append(
                "`mean_ap`/`mean_f1` are omitted from this table: Severstal's test split runs "
                "~92% positive (see the Severstal caveat below), which would make a blended "
                "cross-dataset AP/F1 mean incomparable. AUROC and PG2/PB2 are less "
                "prevalence-sensitive and stay.\n"
            )
        display_ranking = _with_mean_std_strings(_round(ranking, _DISPLAY_DIGITS))
        lines.append(display_ranking.to_markdown(index=False))
        lines.append("")

        table = per_dataset(frame, regime)
        if not table.empty:
            lines.append(f"### Mean AUROC by dataset family -- {regime}\n")
            lines.append(table.to_markdown(index=False))
            lines.append("")

        table = winners(frame, regime)
        if not table.empty:
            lines.append(f"### Best method per configuration -- {regime}\n")
            lines.append(
                _round(
                    table, {"auroc": 4, "average_precision": 4, "f1_max": 4, "ms_per_image": 1}
                ).to_markdown(index=False)
            )
            lines.append("")

    if not failed.empty:
        lines.append("## Failed runs\n")
        lines.append(
            "Recorded rather than dropped: a model that cannot complete a dataset has told "
            "you something about its deployability.\n"
        )
        summary = failed.groupby(["method", "config"])["error"].first().reset_index().head(60)
        lines.append(summary.to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def main_leaderboard(argv: list[str] | None = None) -> None:
    """Parse arguments, summarize the sweep, and write the report."""
    parser = argparse.ArgumentParser(description=main_leaderboard.__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.results_db.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_results(args.results_db)
    regimes = [r for r in ("multiclass", "oneclass") if r in set(frame["regime"])]
    regimes += sorted(set(frame["regime"]) - set(regimes))
    rankings = {regime: rank_regime(frame, regime) for regime in regimes}

    if all(ranking.empty for ranking in rankings.values()):
        msg = "No successful runs to summarize."
        raise SystemExit(msg)

    report = format_report(frame, rankings)
    (output_dir / "leaderboard.md").write_text(report, encoding="utf-8")

    combined = pd.concat([r for r in rankings.values() if not r.empty], ignore_index=True)
    combined.to_csv(output_dir / "ranking.csv", index=False)
    regime_comparison(frame).to_csv(output_dir / "regime_comparison.csv", index=False)
    frame.to_csv(output_dir / "all_runs.csv", index=False)
    for regime in regimes:
        table = winners(frame, regime)
        if not table.empty:
            table.to_csv(output_dir / f"winners_{regime}.csv", index=False)

    print(report)
    print(f"\nWrote leaderboard.md, ranking.csv, regime_comparison.csv to {output_dir}")


# -----------------------------------------------------------------------------
# Subcommand dispatch
# -----------------------------------------------------------------------------

_SUBCOMMANDS = {
    "cost": main_cost,
    "deploy": main_deploy,
    "metrics": main_metrics_backfill,
    "ensemble": main_ensemble,
    "leaderboard": main_leaderboard,
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch to one of the five evaluation stages by subcommand.

    Usage:
        python -m training.evaluate <cost|deploy|metrics|ensemble|leaderboard> [options]
        python -m training.evaluate <stage> --help
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        print(main.__doc__)
        print(f"Stages: {', '.join(_SUBCOMMANDS)}")
        raise SystemExit(0 if not argv else 2)
    _SUBCOMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
