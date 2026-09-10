"""Unit tests for defect localization: the normal-patch bank, the heatmap it
produces, and the detector/pipeline plumbing that carries measurements."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adaptivevision.app import build_metrology_config, build_patch_bank
from adaptivevision.camera import build_frame
from adaptivevision.common import (
    AnomalyResult,
    InferenceEngine,
    RectifiedFrame,
    Verdict,
)
from adaptivevision.config import StationConfig
from adaptivevision.metrology import (
    REGION_NAMES,
    MetrologyConfig,
    NormalPatchBank,
    ThresholdAnomalyDetector,
    heatmap_from_distances,
)
from adaptivevision.orchestration import InspectionPipeline

GRID = 8
PATCHES = GRID * GRID
DIM = 6


def _bank(seed: int = 0) -> NormalPatchBank:
    """A bank of normal patches clustered near the origin."""
    rng = np.random.default_rng(seed)
    return NormalPatchBank(rng.normal(scale=0.1, size=(64, DIM)).astype(np.float32))


def _query_with_outlier(index: int, seed: int = 1) -> np.ndarray:
    """A patch grid matching the bank, except one clearly anomalous patch."""
    rng = np.random.default_rng(seed)
    query = rng.normal(scale=0.1, size=(PATCHES, DIM)).astype(np.float32)
    query[index] += 25.0
    return query


# -----------------------------------------------------------------------------
# NormalPatchBank
# -----------------------------------------------------------------------------


def test_bank_reports_its_shape() -> None:
    bank = _bank()
    assert bank.size == 64
    assert bank.dim == DIM


@pytest.mark.parametrize("bad", [np.zeros((0, DIM), dtype=np.float32), np.zeros(DIM)])
def test_bank_rejects_empty_or_non_2d(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        NormalPatchBank(bad)


def test_distances_are_near_zero_for_normal_patches() -> None:
    """A patch drawn from the same distribution as the bank is not anomalous."""
    bank = _bank()
    rng = np.random.default_rng(7)
    normal = rng.normal(scale=0.1, size=(PATCHES, DIM)).astype(np.float32)
    assert float(bank.distances(normal).max()) < 1.0


def test_distances_spike_on_an_outlier_patch() -> None:
    bank = _bank()
    distances = bank.distances(_query_with_outlier(20))
    assert int(np.argmax(distances)) == 20
    assert float(distances[20]) > 10.0


def test_distances_match_a_brute_force_search() -> None:
    """The matmul identity must agree with an explicit nearest-neighbour scan."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, DIM)).astype(np.float32)
    query = rng.normal(size=(5, DIM)).astype(np.float32)
    expected = np.array(
        [min(float(np.linalg.norm(q - f)) for f in features) for q in query]
    )
    assert np.allclose(NormalPatchBank(features).distances(query), expected, atol=1e-4)


def test_distances_reject_a_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected"):
        _bank().distances(np.zeros((4, DIM + 1), dtype=np.float32))


def test_bank_round_trips_through_disk(tmp_path: Path) -> None:
    bank = _bank()
    path = tmp_path / "nested" / "bank.npz"
    bank.save(path)
    reloaded = NormalPatchBank.load(path)
    assert reloaded.size == bank.size
    query = _query_with_outlier(5)
    assert np.allclose(reloaded.distances(query), bank.distances(query))


def test_load_rejects_an_artifact_without_features(tmp_path: Path) -> None:
    path = tmp_path / "wrong.npz"
    np.savez_compressed(path, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="no 'features'"):
        NormalPatchBank.load(path)


# -----------------------------------------------------------------------------
# Heatmap
# -----------------------------------------------------------------------------


def test_heatmap_locates_the_outlier_patch() -> None:
    """The heatmap must peak where the anomalous patch actually is."""
    bank = _bank()
    # Patch 9 of an 8x8 grid is row 1, col 1 -> upper-left third.
    result = bank.heatmap(_query_with_outlier(9), (64, 64))
    assert result is not None
    heatmap, region = result
    assert heatmap.shape == (64, 64)
    peak_row, peak_col = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    assert peak_row < 32 and peak_col < 32
    assert region == "upper-left"


def test_heatmap_returns_none_for_a_non_square_patch_count() -> None:
    """Degrade rather than guess a grid layout."""
    assert _bank().heatmap(np.zeros((10, DIM), dtype=np.float32), (32, 32)) is None


def test_heatmap_returns_none_for_an_empty_grid() -> None:
    assert _bank().heatmap(np.zeros((0, DIM), dtype=np.float32), (32, 32)) is None


def test_heatmap_of_a_flat_grid_is_all_zero() -> None:
    """A uniform distance field has no peak to normalize against."""
    heatmap, region = heatmap_from_distances(np.full((4, 4), 3.0), (16, 16))
    assert heatmap.max() == 0.0
    assert region in {name for row in REGION_NAMES for name in row}


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [(0, 0, "upper-left"), (1, 1, "center"), (2, 2, "lower-right"), (0, 2, "upper-right")],
)
def test_region_names_map_to_thirds(row: int, col: int, expected: str) -> None:
    grid = np.zeros((3, 3))
    grid[row, col] = 1.0
    _, region = heatmap_from_distances(grid, (9, 9))
    assert region == expected


# -----------------------------------------------------------------------------
# Detector plumbing
# -----------------------------------------------------------------------------


class _FakeEngine(InferenceEngine):
    """Returns a fixed score and patch grid."""

    def __init__(self, score: float, patches: np.ndarray | None) -> None:
        self._score = score
        self._patches = patches

    @property
    def model_version(self) -> str:
        return "fake"

    def load(self, model_id: str) -> None:
        """Unused."""

    def warmup(self) -> None:
        """Unused."""

    def unload(self) -> None:
        """Unused."""

    def infer(self, inputs: Any) -> dict[str, Any]:
        outputs: dict[str, Any] = {"output": np.array([self._score])}
        if self._patches is not None:
            outputs["patch_features"] = self._patches
        return outputs


def _frame() -> RectifiedFrame:
    raw = build_frame(np.zeros((64, 64, 3), dtype=np.uint8), "cam0")
    return RectifiedFrame(
        image=raw.image,
        camera_id=raw.camera_id,
        frame_id=raw.frame_id,
        timestamp_monotonic=raw.timestamp_monotonic,
        timestamp_utc=raw.timestamp_utc,
        calibration_ver="cal-1",
    )


def _config() -> MetrologyConfig:
    return MetrologyConfig(pixel_to_micron=5.0, min_area_px2=4, threshold_percentile=95.0)


def test_detector_measures_defects_on_an_anomalous_frame() -> None:
    detector = ThresholdAnomalyDetector(
        _FakeEngine(0.99, _query_with_outlier(9)),
        threshold=0.5,
        metrology_config=_config(),
        patch_bank=_bank(),
    )
    result = detector.detect(_frame())

    assert result.is_anomalous is True
    assert result.heatmap is not None
    assert result.heatmap_region == "upper-left"
    assert len(result.defect_measurements) >= 1
    largest = result.defect_measurements[0]
    assert largest.area_um2 == pytest.approx(largest.area_px2 * 25.0)
    assert largest.morphology in {"scratch", "particle"}


def test_detector_skips_localization_on_a_passing_frame() -> None:
    """A clean part has no defect to describe, and its normalized heatmap
    would always yield a spurious top-percentile region."""
    detector = ThresholdAnomalyDetector(
        _FakeEngine(0.01, _query_with_outlier(9)),
        threshold=0.5,
        metrology_config=_config(),
        patch_bank=_bank(),
    )
    result = detector.detect(_frame())

    assert result.is_anomalous is False
    assert result.defect_measurements == ()
    assert result.heatmap is None
    assert result.heatmap_region is None


def test_detector_without_a_bank_reports_no_geometry() -> None:
    """Without a normal reference there is nothing to be anomalous against."""
    detector = ThresholdAnomalyDetector(
        _FakeEngine(0.99, _query_with_outlier(9)),
        threshold=0.5,
        metrology_config=_config(),
    )
    result = detector.detect(_frame())

    assert result.is_anomalous is True
    assert result.defect_measurements == ()


def test_detector_without_patch_output_reports_no_geometry() -> None:
    """A model that emits only a score must still produce a verdict."""
    detector = ThresholdAnomalyDetector(
        _FakeEngine(0.99, None),
        threshold=0.5,
        metrology_config=_config(),
        patch_bank=_bank(),
    )
    assert detector.detect(_frame()).defect_measurements == ()


def test_detector_verdict_is_unaffected_by_localization() -> None:
    """Localization describes; it never decides."""
    engine = _FakeEngine(0.6, _query_with_outlier(9))
    plain = ThresholdAnomalyDetector(engine, threshold=0.5).detect(_frame())
    localized = ThresholdAnomalyDetector(
        engine, threshold=0.5, metrology_config=_config(), patch_bank=_bank()
    ).detect(_frame())

    assert plain.is_anomalous == localized.is_anomalous
    assert plain.score == localized.score
    assert plain.defects == localized.defects


# -----------------------------------------------------------------------------
# Pipeline propagation
# -----------------------------------------------------------------------------


class _FixedCamera:
    """Camera returning one blank RGB frame."""

    def open(self) -> None:
        """Unused."""

    def close(self) -> None:
        """Unused."""

    def is_healthy(self) -> bool:
        return True

    def capture(self, trigger_id: str | None = None) -> Any:
        return build_frame(np.zeros((64, 64, 3), dtype=np.uint8), "cam0", trigger_id=trigger_id)


class _StubDetector:
    """Anomaly detector returning fixed measurements."""

    def __init__(self, result: AnomalyResult) -> None:
        self._result = result

    def detect(self, frame: Any, roi: Any = None) -> AnomalyResult:
        return self._result


def test_pipeline_carries_measurements_and_region_into_the_result() -> None:
    """The measurements the detector took must reach the durable record --
    this is what the advisory layer and the MES row are built from."""
    bank = _bank()
    localized = ThresholdAnomalyDetector(
        _FakeEngine(0.99, _query_with_outlier(9)),
        threshold=0.5,
        metrology_config=_config(),
        patch_bank=bank,
    ).detect(_frame())

    pipeline = InspectionPipeline(
        _FixedCamera(),  # type: ignore[arg-type]
        station_id="s1",
        recipe_ver="r1",
        anomaly_detector=_StubDetector(localized),  # type: ignore[arg-type]
        lot_id="LOT-TEST",
    )
    result = pipeline.run("part-1")

    assert result.verdict is Verdict.FAIL
    assert result.lot_id == "LOT-TEST"
    assert result.heatmap_region == localized.heatmap_region
    assert result.defect_measurements == localized.defect_measurements
    assert result.defect_measurements[0].area_um2 > 0


def test_result_round_trips_measurements_and_region() -> None:
    from adaptivevision.common import InspectionResult

    original = InspectionResult(
        inspection_id="i1",
        part_id="p1",
        station_id="s1",
        verdict=Verdict.FAIL,
        recipe_ver="r",
        model_ver="m",
        calib_ver="c",
        cycle_time_ms=1.0,
        timestamp_utc=datetime.now(UTC),
        heatmap_region="center",
    )
    assert InspectionResult.from_dict(original.to_dict()) == original


def test_anomaly_result_does_not_serialize_the_heatmap_array() -> None:
    """The array is transient working data; the record keeps the archived
    image and the measurements derived from it."""
    result = AnomalyResult(
        score=0.9,
        threshold=0.5,
        is_anomalous=True,
        heatmap_region="center",
        heatmap=np.zeros((4, 4)),
    )
    payload = result.to_dict()
    assert "heatmap" not in payload
    assert payload["heatmap_region"] == "center"


# -----------------------------------------------------------------------------
# Composition root
# -----------------------------------------------------------------------------


def _station(**extra: object) -> StationConfig:
    return StationConfig(station_id="s1", log_level="INFO", extra=dict(extra))


def test_metrology_config_absent_unless_enabled() -> None:
    assert build_metrology_config(_station()) is None
    assert build_metrology_config(_station(METROLOGY_ENABLED="false")) is None


def test_metrology_config_when_enabled() -> None:
    config = build_metrology_config(_station(METROLOGY_ENABLED="true"))
    assert config is not None
    assert config.pixel_to_micron > 0


def test_patch_bank_absent_without_a_path() -> None:
    assert build_patch_bank(_station()) is None


def test_patch_bank_loads_from_configured_path(tmp_path: Path) -> None:
    path = tmp_path / "bank.npz"
    _bank().save(path)
    bank = build_patch_bank(_station(PATCH_BANK_PATH=str(path)))
    assert bank is not None
    assert bank.size == 64


def test_missing_patch_bank_degrades_instead_of_failing_boot(tmp_path: Path) -> None:
    """A missing artifact disables localization; it never stops the station."""
    assert build_patch_bank(_station(PATCH_BANK_PATH=str(tmp_path / "absent.npz"))) is None
