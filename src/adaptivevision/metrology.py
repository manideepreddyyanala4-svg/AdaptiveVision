"""Analyze stage, part 2: score the part and measure the defect.

Three related concerns: AI anomaly detection (score a frame with a model and
apply a decision threshold), heatmap-derived defect metrology (Milestone
M21 -- contour segmentation, bounding box, physical area, morphology), and
classical dimensional metrology (evaluate recipe measurement specs against
measured feature values). All three produce a :class:`~adaptivevision.common.PartialResult`
consumed by ``decision.py``.

The metrology functions below have no OpenCV dependency, matching this
package's established convention of keeping ``src/adaptivevision`` dependency
-free for numerical routines it can reasonably implement itself:
:func:`otsu_threshold` is a from-scratch, 256-bin histogram implementation of
Otsu's method, and :func:`_label_connected_components` is a from-scratch
4-connectivity flood-fill labeler. Heatmaps here are small (one uploaded/
inspected image at a time, not a video stream), so the per-pixel Python loop
is not a bottleneck.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptivevision.camera import LocalizedPart
from adaptivevision.common import (
    ROI,
    AnomalyDetector,
    AnomalyResult,
    DefectClass,
    DefectMeasurement,
    InferenceEngine,
    Inspector,
    Measurement,
    MetrologyResult,
    RectifiedFrame,
    Severity,
)
from adaptivevision.common import Defect as _Defect

Defect = _Defect

_NDArray = np.ndarray[Any, np.dtype[Any]]

# =============================================================================
# AI anomaly detection
#
# Concrete AnomalyDetector implementations that score a rectified frame and
# produce an AnomalyResult. A deterministic static detector is provided for
# replay, tests, and simulated stations; a threshold detector wraps an
# InferenceEngine so a real model can be used without coupling the inspector
# to ONNX specifically.
# =============================================================================

#: Extracts a scalar anomaly score from an inference output mapping.
ScoreExtractor = Callable[[Mapping[str, _NDArray]], float]


class StaticAnomalyDetector(AnomalyDetector):
    """Deterministic anomaly detector for replay, tests, and simulated stations.

    Args:
        score: Fixed anomaly score returned for every frame.
        threshold: Decision threshold the score is compared against.
        heatmap_ref: Optional heatmap reference to attach to the result.
    """

    def __init__(
        self,
        score: float,
        threshold: float,
        *,
        heatmap_ref: str | None = None,
    ) -> None:
        """Initialize the detector."""
        self._score = score
        self._threshold = threshold
        self._heatmap_ref = heatmap_ref

    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        """Return a fixed anomaly result for the frame.

        Args:
            frame: The rectified frame to analyze.
            roi: Optional region to restrict analysis to.

        Returns:
            An :class:`~adaptivevision.common.AnomalyResult` with the
            configured score and threshold.
        """
        _ = (frame, roi)
        is_anomalous = self._score >= self._threshold
        defects: tuple[Defect, ...] = ()
        if is_anomalous:
            defects = (
                Defect(
                    defect_class=DefectClass.ANOMALY,
                    severity=Severity.MAJOR,
                    score=self._score,
                    roi=roi,
                    description="Anomaly score exceeded threshold",
                ),
            )
        return AnomalyResult(
            score=self._score,
            threshold=self._threshold,
            is_anomalous=is_anomalous,
            heatmap_ref=self._heatmap_ref,
            defects=defects,
        )


class ThresholdAnomalyDetector(AnomalyDetector):
    """Score a frame with an inference engine and apply a decision threshold.

    Args:
        engine: The inference engine used to score the frame.
        threshold: Anomaly score threshold; scores at or above it are anomalous.
        input_name: Name of the model input the frame image is fed to.
        output_name: Name of the model output that carries the anomaly score.
        score_extractor: Optional callable that maps the inference output to a
            scalar score. Defaults to reading ``output_name`` and taking the
            first element.
        anomalous_severity: Severity assigned to the defect raised for an
            anomalous frame. A recipe's ``review_on_anomaly`` flag maps to
            ``Severity.MINOR`` here (the decision policy routes MINOR to
            REVIEW); the default ``Severity.MAJOR`` routes straight to FAIL.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        threshold: float,
        *,
        input_name: str = "input",
        output_name: str = "output",
        score_extractor: ScoreExtractor | None = None,
        anomalous_severity: Severity = Severity.MAJOR,
    ) -> None:
        """Initialize the detector."""
        self._engine = engine
        self._threshold = threshold
        self._input_name = input_name
        self._output_name = output_name
        self._score_extractor = score_extractor or self._default_score
        self._anomalous_severity = anomalous_severity

    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        """Score the frame and apply the decision threshold.

        Args:
            frame: The rectified frame to analyze.
            roi: Optional region to restrict analysis to.

        Returns:
            An :class:`~adaptivevision.common.AnomalyResult` with the model
            score and threshold.

        Raises:
            InferenceError: If inference fails.
        """
        _ = roi
        image = frame.image.astype(np.float32, copy=False)
        if image.ndim == 2:
            # Grayscale (H, W) -> (1, H, W): a 1-channel model input.
            image = image[np.newaxis, ...]
        elif image.ndim == 3:
            # Color (H, W, C) -> (C, H, W): the channel-first layout ONNX
            # vision models expect.
            image = image.transpose(2, 0, 1)
        outputs = self._engine.infer({self._input_name: image})
        score = self._score_extractor(outputs)
        is_anomalous = score >= self._threshold
        defects: tuple[Defect, ...] = ()
        if is_anomalous:
            defects = (
                Defect(
                    defect_class=DefectClass.ANOMALY,
                    severity=self._anomalous_severity,
                    score=score,
                    description="Anomaly score exceeded threshold",
                ),
            )
        return AnomalyResult(
            score=score,
            threshold=self._threshold,
            is_anomalous=is_anomalous,
            defects=defects,
        )

    @staticmethod
    def _default_score(outputs: Mapping[str, _NDArray]) -> float:
        """Extract a scalar score from the configured output tensor."""
        values = outputs["output"]
        return float(np.asarray(values).reshape(-1)[0])


# =============================================================================
# Defect metrology (Milestone M21)
#
# Thresholds a 2D anomaly heatmap into a binary defect mask, segments it into
# connected regions, and measures each one: pixel-space bounding box,
# physical area in square microns (via a caller-supplied pixel-to-micron
# calibration factor), and a coarse morphology classification -- a long, thin
# region reads as a scratch, a roughly round one as a particle/void.
# =============================================================================

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
        One :class:`~adaptivevision.common.DefectMeasurement` per connected
        foreground region at or above ``config.min_area_px2``, largest
        first. Empty when the heatmap has no defect region (or everything
        above threshold is smaller than the noise floor).

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


# =============================================================================
# Classical dimensional metrology
#
# Evaluate recipe measurement specs against measured feature values. A
# deterministic static source is provided for replay/tests/simulated
# stations; a real source (a laser micrometer, a vision caliper) can replace
# it behind the same callable boundary without changing this inspector.
# =============================================================================

MeasurementSource = Callable[[LocalizedPart, Any], Mapping[str, float]]


class StaticMeasurementSource:
    """Deterministic measurement source for replay/tests/simulated stations."""

    def __init__(self, values: Mapping[str, float]) -> None:
        """Initialize with measured values keyed by measurement spec name."""
        self._values = dict(values)

    def measure(self, part: LocalizedPart, recipe: Any) -> Mapping[str, float]:
        """Return the configured measurements.

        Args:
            part: Localized part, accepted for interface compatibility.
            recipe: Active recipe, accepted for interface compatibility.
        """
        _ = (part, recipe)
        return dict(self._values)


class MetrologyInspector(Inspector[LocalizedPart, Any]):
    """Evaluate recipe measurement specs against measured feature values.

    Args:
        source: Callable that produces measured values in the units declared
            by each :class:`~adaptivevision.common.MeasurementSpec`.
    """

    def __init__(self, source: MeasurementSource) -> None:
        """Initialize the inspector."""
        self._source = source

    def inspect(self, part: LocalizedPart, recipe: Any) -> MetrologyResult:
        """Inspect an aligned part against recipe measurement specs."""
        values = self._source(part, recipe)
        measurements: list[Measurement] = []
        defects: list[Defect] = []

        for spec in recipe.measurement_specs:
            if spec.name not in values:
                defects.append(
                    Defect(
                        defect_class=DefectClass.DIMENSIONAL,
                        severity=Severity.CRITICAL,
                        description=f"Missing measurement for spec {spec.name!r}",
                    )
                )
                continue

            value = values[spec.name]
            in_tolerance = spec.contains(value)
            measurement = Measurement(
                name=spec.name,
                value=value,
                unit=spec.unit,
                spec=spec,
                in_tolerance=in_tolerance,
            )
            measurements.append(measurement)
            if not in_tolerance:
                defects.append(
                    Defect(
                        defect_class=DefectClass.DIMENSIONAL,
                        severity=Severity.MAJOR,
                        description=(
                            f"Measurement {spec.name!r}={value} {spec.unit} outside tolerance"
                        ),
                    )
                )

        return MetrologyResult(measurements=tuple(measurements), defects=tuple(defects))
