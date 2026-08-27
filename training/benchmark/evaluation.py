"""Metrics shared by every method in the zoo.

AUROC ranks models but does not tell you what a line would actually cost. A
station has to commit to one threshold, and the two errors it can make have
completely different prices: scrapping a good part wastes material, shipping a
defective one reaches a customer. So alongside the ranking metrics this module
reports both error rates at fixed, interpretable operating points:

* **FPR@95TPR** -- the over-rejection (scrap) rate if you insist on catching
  95% of defects.
* **FNR@1FPR** -- the escape rate if you can only tolerate 1% false alarms.

Both halves of the zoo go through this module on raw scores, rather than each
backend reporting its own metric objects, or the numbers would not be
comparable.

Localization is scored separately where ground-truth masks exist. Pixel AUROC
alone flatters a model that lights up a large blob near the defect, so AUPRO
(per-region overlap, integrated to 30% FPR) is reported with it -- it weights
every defect region equally regardless of size, which is what matters when the
small defects are the expensive ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

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
    ``tests/test_evaluation.py``). Treat it as a best-effort translation of
    the definition above, not a verified-bit-exact port.

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
