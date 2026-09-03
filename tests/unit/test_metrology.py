"""Unit tests for metrology.py: anomaly detection and defect metrology."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from adaptivevision.common import (
    ROI,
    AnomalyResult,
    DefectClass,
    DefectMeasurement,
    InferenceEngine,
    RectifiedFrame,
    Severity,
)
from adaptivevision.metrology import (
    PARTICLE,
    SCRATCH,
    MetrologyConfig,
    StaticAnomalyDetector,
    ThresholdAnomalyDetector,
    measure_defects,
    otsu_threshold,
)

# -----------------------------------------------------------------------------
# AI anomaly detection
# -----------------------------------------------------------------------------

def _frame(image: np.ndarray[Any, np.dtype[Any]]) -> RectifiedFrame:
    return RectifiedFrame(
        image=image,
        camera_id="cam-1",
        frame_id="frame-1",
        calibration_ver="cal-1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime.now(UTC),
    )


class _FakeEngine(InferenceEngine):
    def __init__(self, score: float) -> None:
        self._score = score
        self._inputs: list[dict[str, np.ndarray[Any, np.dtype[Any]]]] = []

    @property
    def model_version(self) -> str:
        return "model-v1"

    def load(self, model_id: str) -> None:
        _ = model_id

    def warmup(self) -> None:
        return None

    def infer(
        self, inputs: dict[str, np.ndarray[Any, np.dtype[Any]]]
    ) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
        self._inputs.append(inputs)
        return {"output": np.array([self._score], dtype=np.float32)}

    def unload(self) -> None:
        return None


def test_static_detector_below_threshold_is_not_anomalous() -> None:
    detector = StaticAnomalyDetector(score=0.2, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.score == 0.2
    assert result.threshold == 0.5
    assert result.is_anomalous is False
    assert result.defects == ()


def test_static_detector_at_threshold_is_anomalous_with_defect() -> None:
    detector = StaticAnomalyDetector(score=0.5, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.is_anomalous is True
    assert len(result.defects) == 1
    defect = result.defects[0]
    assert defect.defect_class == DefectClass.ANOMALY
    assert defect.severity == Severity.MAJOR
    assert defect.score == 0.5


def test_static_detector_carries_heatmap_ref_and_roi() -> None:
    roi = ROI(label="region", x=0.0, y=0.0, width=2.0, height=2.0)
    detector = StaticAnomalyDetector(
        score=0.9, threshold=0.5, heatmap_ref="heatmaps/part-1.png"
    )
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)), roi=roi)
    assert result.heatmap_ref == "heatmaps/part-1.png"
    assert result.defects[0].roi == roi


def test_threshold_detector_batches_2d_image_and_extracts_score() -> None:
    engine = _FakeEngine(score=0.8)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.score == pytest.approx(0.8)
    assert result.is_anomalous is True
    assert engine._inputs[0]["input"].ndim == 3


def test_threshold_detector_below_threshold_is_clean() -> None:
    engine = _FakeEngine(score=0.1)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.is_anomalous is False
    assert result.defects == ()


def test_threshold_detector_uses_custom_score_extractor() -> None:
    engine = _FakeEngine(score=0.0)
    detector = ThresholdAnomalyDetector(
        engine,
        threshold=0.5,
        score_extractor=lambda outputs: float(outputs["output"][0]) + 0.9,
    )
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.score == pytest.approx(0.9)
    assert result.is_anomalous is True


def test_anomaly_result_round_trips() -> None:
    result = AnomalyResult(
        score=0.7,
        threshold=0.5,
        is_anomalous=True,
        heatmap_ref="heatmaps/part-1.png",
    )
    restored = AnomalyResult.from_dict(result.to_dict())
    assert restored == result


# -----------------------------------------------------------------------------
# Defect metrology
# -----------------------------------------------------------------------------

def test_otsu_threshold_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        otsu_threshold(np.array([]))


def test_otsu_threshold_constant_array_returns_that_value() -> None:
    values = np.full((10, 10), 5.0)
    assert otsu_threshold(values) == 5.0


def test_otsu_threshold_separates_two_clear_clusters() -> None:
    low = np.zeros((50,))
    high = np.full((50,), 10.0)
    values = np.concatenate([low, high])
    threshold = otsu_threshold(values)
    assert 0.0 < threshold < 10.0


def test_metrology_config_rejects_non_positive_pixel_to_micron() -> None:
    with pytest.raises(ValueError, match="pixel_to_micron"):
        MetrologyConfig(pixel_to_micron=0.0)
    with pytest.raises(ValueError, match="pixel_to_micron"):
        MetrologyConfig(pixel_to_micron=-1.0)


def test_metrology_config_rejects_out_of_range_percentile() -> None:
    with pytest.raises(ValueError, match="threshold_percentile"):
        MetrologyConfig(pixel_to_micron=1.0, threshold_percentile=101.0)


def test_metrology_config_rejects_non_positive_min_area() -> None:
    with pytest.raises(ValueError, match="min_area_px2"):
        MetrologyConfig(pixel_to_micron=1.0, min_area_px2=0)


def test_measure_defects_rejects_non_2d_heatmap() -> None:
    config = MetrologyConfig(pixel_to_micron=1.0)
    with pytest.raises(ValueError, match="2D"):
        measure_defects(np.zeros((4, 4, 3)), config)


def test_measure_defects_flat_heatmap_finds_nothing() -> None:
    heatmap = np.zeros((32, 32))
    config = MetrologyConfig(pixel_to_micron=1.0)
    assert measure_defects(heatmap, config) == []


def _round_particle_heatmap() -> np.ndarray:
    heatmap = np.zeros((32, 32))
    heatmap[14:18, 14:18] = 1.0  # 4x4 roughly square blob
    return heatmap


def _elongated_scratch_heatmap() -> np.ndarray:
    heatmap = np.zeros((32, 32))
    heatmap[15:17, 2:26] = 1.0  # 2 tall x 24 wide -- clearly a scratch
    return heatmap


def test_measure_defects_classifies_round_region_as_particle() -> None:
    config = MetrologyConfig(pixel_to_micron=2.0)
    measurements = measure_defects(_round_particle_heatmap(), config)
    assert len(measurements) == 1
    measurement = measurements[0]
    assert isinstance(measurement, DefectMeasurement)
    assert measurement.morphology == PARTICLE
    assert measurement.bbox == (14, 14, 4, 4)
    assert measurement.area_px2 == 16
    assert measurement.area_um2 == pytest.approx(16 * 2.0**2)
    assert measurement.aspect_ratio == pytest.approx(1.0)


def test_measure_defects_classifies_elongated_region_as_scratch() -> None:
    config = MetrologyConfig(pixel_to_micron=1.0)
    measurements = measure_defects(_elongated_scratch_heatmap(), config)
    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.morphology == SCRATCH
    assert measurement.bbox == (2, 15, 24, 2)
    assert measurement.area_px2 == 48
    assert measurement.aspect_ratio == pytest.approx(12.0)


def test_measure_defects_finds_multiple_disjoint_regions_largest_first() -> None:
    heatmap = np.zeros((40, 40))
    heatmap[2:4, 2:4] = 1.0  # small 2x2 = 4px
    heatmap[20:30, 20:26] = 1.0  # big 10x6 = 60px
    config = MetrologyConfig(pixel_to_micron=1.0)

    measurements = measure_defects(heatmap, config)

    assert len(measurements) == 2
    assert measurements[0].area_px2 == 60
    assert measurements[1].area_px2 == 4


def test_measure_defects_discards_regions_smaller_than_min_area() -> None:
    heatmap = np.zeros((20, 20))
    heatmap[5:6, 5:6] = 1.0  # a single pixel
    config = MetrologyConfig(pixel_to_micron=1.0, min_area_px2=4)
    assert measure_defects(heatmap, config) == []


def test_measure_defects_percentile_threshold_overrides_otsu() -> None:
    heatmap = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    config = MetrologyConfig(pixel_to_micron=1.0, threshold_percentile=99.0, min_area_px2=1)
    measurements = measure_defects(heatmap, config)
    # Only the very top percentile of a smooth gradient should survive.
    total_flagged = sum(m.area_px2 for m in measurements)
    assert 0 < total_flagged <= 2


def test_defect_measurement_roundtrip() -> None:
    original = DefectMeasurement(
        bbox=(1, 2, 3, 4),
        area_px2=12,
        area_um2=48.0,
        aspect_ratio=1.333,
        morphology=SCRATCH,
    )
    restored = DefectMeasurement.from_dict(original.to_dict())
    assert restored == original
