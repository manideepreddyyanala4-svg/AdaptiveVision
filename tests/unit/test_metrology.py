"""Unit tests for M21 defect metrology (heatmap-derived shape measurement)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptivevision.common.result import DefectMeasurement
from adaptivevision.inspection.anomaly.metrology import (
    PARTICLE,
    SCRATCH,
    MetrologyConfig,
    measure_defects,
    otsu_threshold,
)


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
