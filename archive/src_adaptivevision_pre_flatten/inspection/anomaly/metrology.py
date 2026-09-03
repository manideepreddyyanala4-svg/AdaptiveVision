"""Defect metrology: contour-derived shape measurements from an anomaly
heatmap (Milestone M21).

An anomaly score alone tells a line "something's wrong here"; a semiconductor
or precision-parts fab needs to know how big, what shape, and where. This
module thresholds a 2D anomaly heatmap into a binary defect mask, segments it
into connected regions, and measures each one: pixel-space bounding box,
physical area in square microns (via a caller-supplied pixel-to-micron
calibration factor), and a coarse morphology classification -- a long, thin
region reads as a **scratch**, a roughly round one as a **particle/void**.

No OpenCV dependency, matching this package's established convention (see
``monitoring/spc.py``, ``preprocessing/operators.py::resize_to``,
``monitoring/drift.py``) of keeping ``src/adaptivevision`` dependency-free for
numerical routines it can reasonably implement itself:
:func:`otsu_threshold` is a from-scratch, 256-bin histogram implementation of
Otsu's method (the same binning OpenCV's own 8-bit ``THRESH_OTSU`` uses), and
:func:`_label_connected_components` is a from-scratch 4-connectivity
flood-fill labeler (the pure-Python equivalent of
``cv2.connectedComponents``/``scipy.ndimage.label``). Heatmaps here are small
(one uploaded/inspected image at a time, not a video stream), so the
per-pixel Python loop is not a bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: This module's numpy arrays are always real-valued (scores) or boolean
#: (masks/labels); the dtype parameter carries no useful static information
#: here, so every signature below spells it out as ``Any`` for mypy --strict
#: rather than repeating a dtype union that adds noise without catching bugs.
_NDArray = np.ndarray[Any, np.dtype[Any]]

from adaptivevision.common.result import DefectMeasurement

#: Morphology classification for an elongated region (length/width > threshold).
SCRATCH = "scratch"

#: Morphology classification for a roughly round region.
PARTICLE = "particle"

#: Aspect ratio above which a region is classified as a scratch rather than a
#: particle/void. Matches the conventional AOI heuristic that a defect more
#: than 3x longer than it is wide reads as a linear/scratch-like feature.
_SCRATCH_ASPECT_RATIO_THRESHOLD = 3.0

#: Regions smaller than this many pixels are discarded as heatmap noise
#: rather than reported as defects.
_DEFAULT_MIN_AREA_PX2 = 4

#: Bin count for the Otsu histogram. 256 matches OpenCV's own 8-bit
#: THRESH_OTSU behavior and is more than enough resolution for a heatmap.
_OTSU_BINS = 256


def otsu_threshold(values: _NDArray) -> float:
    """Compute Otsu's threshold: the cut point maximizing between-class variance.

    Args:
        values: Array of real-valued scores (any shape; flattened internally).

    Returns:
        The threshold value, on the same scale as ``values``. Returns the
        array's minimum when every value is identical (nothing to split).

    Raises:
        ValueError: If ``values`` is empty.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        msg = "otsu_threshold requires a non-empty array"
        raise ValueError(msg)

    low, high = float(flat.min()), float(flat.max())
    if high <= low:
        return low

    histogram, bin_edges = np.histogram(flat, bins=_OTSU_BINS, range=(low, high))
    weights = histogram.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight_below = np.cumsum(weights)
    weight_above = weights.sum() - weight_below
    sum_below = np.cumsum(weights * bin_centers)
    total_sum = sum_below[-1]

    mean_below = np.divide(
        sum_below, weight_below, out=np.zeros_like(sum_below), where=weight_below > 0
    )
    mean_above = np.divide(
        total_sum - sum_below,
        weight_above,
        out=np.zeros_like(sum_below),
        where=weight_above > 0,
    )
    between_class_variance = weight_below * weight_above * (mean_below - mean_above) ** 2

    best_bin = int(np.argmax(between_class_variance))
    return float(bin_centers[best_bin])


def _label_connected_components(mask: _NDArray) -> tuple[_NDArray, int]:
    """Label 4-connected regions of ``True`` pixels via iterative flood fill.

    Args:
        mask: 2D boolean array, ``True`` marking foreground pixels.

    Returns:
        ``(labels, count)``: an ``int32`` array the same shape as ``mask``
        with ``0`` for background and ``1..count`` per region (first-seen
        order), and the number of regions found.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 0

    for start_row in range(height):
        for start_col in range(width):
            if not mask[start_row, start_col] or labels[start_row, start_col] != 0:
                continue
            current_label += 1
            labels[start_row, start_col] = current_label
            stack = [(start_row, start_col)]
            while stack:
                row, col = stack.pop()
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row, next_col = row + d_row, col + d_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and labels[next_row, next_col] == 0
                    ):
                        labels[next_row, next_col] = current_label
                        stack.append((next_row, next_col))

    return labels, current_label


@dataclass(frozen=True, slots=True)
class MetrologyConfig:
    """Configuration for :func:`measure_defects`.

    Attributes:
        pixel_to_micron: Physical length, in microns, of one heatmap pixel's
            edge (assumes square pixels). Must be positive.
        threshold_percentile: When set, threshold the heatmap at this
            percentile (0-100) of its own values instead of computing an
            Otsu threshold -- useful when a heatmap's value distribution
            isn't the roughly bimodal shape Otsu's method assumes (e.g. a
            mostly-flat map with one small hot region).
        min_area_px2: Regions smaller than this are discarded as noise.
    """

    pixel_to_micron: float
    threshold_percentile: float | None = None
    min_area_px2: int = _DEFAULT_MIN_AREA_PX2

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.pixel_to_micron <= 0:
            msg = "pixel_to_micron must be positive"
            raise ValueError(msg)
        if self.threshold_percentile is not None and not 0.0 <= self.threshold_percentile <= 100.0:
            msg = "threshold_percentile must be in [0, 100]"
            raise ValueError(msg)
        if self.min_area_px2 < 1:
            msg = "min_area_px2 must be at least 1"
            raise ValueError(msg)


def measure_defects(heatmap: _NDArray, config: MetrologyConfig) -> list[DefectMeasurement]:
    """Threshold, segment, and measure defect regions in an anomaly heatmap.

    Args:
        heatmap: 2D array of per-pixel anomaly scores, higher meaning more
            anomalous.
        config: Calibration and thresholding configuration.

    Returns:
        One :class:`~adaptivevision.common.result.DefectMeasurement` per
        connected foreground region at or above ``config.min_area_px2``,
        largest first. Empty when the heatmap has no defect region (or
        everything above threshold is smaller than the noise floor).

    Raises:
        ValueError: If ``heatmap`` is not 2-dimensional.
    """
    if heatmap.ndim != 2:
        msg = f"heatmap must be a 2D array, got shape {heatmap.shape!r}"
        raise ValueError(msg)

    # A perfectly flat heatmap has no foreground/background split to find --
    # Otsu's threshold degenerates to the array's own floor in that case
    # (nothing to maximize between-class variance over), which would
    # otherwise flag the entire image as one giant "defect" via `>=`.
    if float(heatmap.max()) <= float(heatmap.min()):
        return []

    if config.threshold_percentile is not None:
        threshold = float(np.percentile(heatmap, config.threshold_percentile))
    else:
        threshold = otsu_threshold(heatmap)

    mask = heatmap >= threshold
    labels, count = _label_connected_components(mask)

    measurements: list[DefectMeasurement] = []
    for region_id in range(1, count + 1):
        rows, cols = np.nonzero(labels == region_id)
        area_px2 = int(rows.size)
        if area_px2 < config.min_area_px2:
            continue

        x0, x1 = int(cols.min()), int(cols.max())
        y0, y1 = int(rows.min()), int(rows.max())
        box_width = x1 - x0 + 1
        box_height = y1 - y0 + 1
        long_side = max(box_width, box_height)
        short_side = max(1, min(box_width, box_height))
        aspect_ratio = long_side / short_side
        morphology = SCRATCH if aspect_ratio > _SCRATCH_ASPECT_RATIO_THRESHOLD else PARTICLE

        measurements.append(
            DefectMeasurement(
                bbox=(x0, y0, box_width, box_height),
                area_px2=area_px2,
                area_um2=area_px2 * (config.pixel_to_micron**2),
                aspect_ratio=aspect_ratio,
                morphology=morphology,
            )
        )

    measurements.sort(key=lambda m: m.area_px2, reverse=True)
    return measurements
