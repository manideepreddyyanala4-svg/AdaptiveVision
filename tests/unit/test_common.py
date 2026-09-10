"""Unit tests for common.py: enums, errors, geometry, IDs, timing, value
types, result shapes, and the abstraction seams."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from adaptivevision import common as enums
from adaptivevision import common as errors
from adaptivevision import common as geometry
from adaptivevision import common as ids
from adaptivevision import common as interfaces
from adaptivevision import common as result
from adaptivevision import common as timing
from adaptivevision import common as types
from adaptivevision.common import (
    ROI,
    AnomalyResult,
    DefectClass,
    InspectionResult,
    MetrologyResult,
    RawFrame,
    RectifiedFrame,
    Severity,
    Verdict,
)

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

def test_verdict_values_are_pinned() -> None:
    assert enums.Verdict.PASS.value == "pass"
    assert enums.Verdict.FAIL.value == "fail"
    assert enums.Verdict.REVIEW.value == "review"


def test_verdict_is_string() -> None:
    assert enums.Verdict.PASS == "pass"
    assert isinstance(enums.Verdict.PASS, str)


def test_severity_values_are_pinned() -> None:
    assert [s.value for s in enums.Severity] == ["info", "minor", "major", "critical"]


def test_defect_class_values_are_pinned() -> None:
    assert enums.DefectClass.DIMENSIONAL.value == "dimensional"
    assert enums.DefectClass.ANOMALY.value == "anomaly"
    assert enums.DefectClass.UNKNOWN.value == "unknown"


def test_station_state_covers_spec_states() -> None:
    values = {s.value for s in enums.StationState}
    assert {"init", "running", "fault", "estop", "shutdown"} <= values


def test_roundtrip_from_value() -> None:
    for member in enums.CameraKind:
        assert enums.CameraKind(member.value) is member


def test_execution_provider_order() -> None:
    assert [p.value for p in enums.ExecutionProvider] == [
        "tensorrt",
        "cuda",
        "openvino",
        "cpu",
    ]


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------

def test_all_errors_derive_from_base() -> None:
    for cls in (
        errors.AcquisitionError,
        errors.CalibrationError,
        errors.InferenceError,
        errors.CommsError,
        errors.RecipeError,
        errors.FaultError,
    ):
        assert issubclass(cls, errors.AdaptiveVisionError)


def test_recoverable_defaults() -> None:
    assert errors.AcquisitionError("x").recoverable is True
    assert errors.InferenceError("x").recoverable is True
    assert errors.CommsError("x").recoverable is True
    assert errors.CalibrationError("x").recoverable is False
    assert errors.RecipeError("x").recoverable is False
    assert errors.FaultError("x").recoverable is False


def test_recoverable_override() -> None:
    err = errors.CalibrationError("x", recoverable=True)
    assert err.recoverable is True
    assert err.is_fatal is False


def test_is_fatal_is_negation_of_recoverable() -> None:
    assert errors.FaultError("x").is_fatal is True
    assert errors.AcquisitionError("x").is_fatal is False


def test_message_is_preserved() -> None:
    err = errors.CommsError("broker down")
    assert err.message == "broker down"
    assert str(err) == "broker down"


def test_can_be_raised_and_caught_as_base() -> None:
    with pytest.raises(errors.AdaptiveVisionError):
        raise errors.RecipeError("bad recipe")


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def test_deg_rad_roundtrip() -> None:
    assert geometry.deg_to_rad(180.0) == pytest.approx(math.pi)
    assert geometry.rad_to_deg(math.pi) == pytest.approx(180.0)


def test_normalize_angle_deg() -> None:
    assert geometry.normalize_angle_deg(190.0) == pytest.approx(-170.0)
    assert geometry.normalize_angle_deg(-190.0) == pytest.approx(170.0)
    assert geometry.normalize_angle_deg(0.0) == pytest.approx(0.0)
    assert geometry.normalize_angle_deg(180.0) == pytest.approx(-180.0)


def test_distance() -> None:
    assert geometry.distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_angle_between_deg() -> None:
    assert geometry.angle_between_deg((0.0, 0.0), (1.0, 1.0)) == pytest.approx(45.0)


def test_rotate_point_about_origin() -> None:
    x, y = geometry.rotate_point((1.0, 0.0), 90.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(1.0)


def test_rotate_point_about_custom_origin() -> None:
    x, y = geometry.rotate_point((2.0, 1.0), 90.0, origin=(1.0, 1.0))
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)


def test_translate_point() -> None:
    assert geometry.translate_point((1.0, 2.0), 3.0, -1.0) == (4.0, 1.0)


def test_transform_point_rotate_then_translate() -> None:
    x, y = geometry.transform_point((10.0, 5.0, 90.0), (1.0, 0.0))
    assert x == pytest.approx(10.0, abs=1e-9)
    assert y == pytest.approx(6.0)


def test_compose_matches_sequential_transform() -> None:
    first = (1.0, 2.0, 30.0)
    second = (-3.0, 4.0, 45.0)
    point = (5.0, -2.0)
    composed = geometry.compose_pose(first, second)
    via_compose = geometry.transform_point(composed, point)
    via_sequential = geometry.transform_point(
        second, geometry.transform_point(first, point)
    )
    assert via_compose[0] == pytest.approx(via_sequential[0])
    assert via_compose[1] == pytest.approx(via_sequential[1])


def test_invert_pose_is_inverse() -> None:
    pose = (3.0, -1.5, 37.0)
    identity = geometry.compose_pose(pose, geometry.invert_pose(pose))
    assert identity[0] == pytest.approx(0.0, abs=1e-9)
    assert identity[1] == pytest.approx(0.0, abs=1e-9)
    assert identity[2] == pytest.approx(0.0, abs=1e-9)


def test_apply_homography_identity() -> None:
    identity: geometry.Homography = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert geometry.apply_homography(identity, (2.0, 3.0)) == pytest.approx((2.0, 3.0))


def test_apply_homography_scaling() -> None:
    scale: geometry.Homography = (
        (2.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    x, y = geometry.apply_homography(scale, (4.0, 5.0))
    assert x == pytest.approx(8.0)
    assert y == pytest.approx(15.0)


def test_apply_homography_raises_on_degenerate() -> None:
    degenerate: geometry.Homography = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="Degenerate homography"):
        geometry.apply_homography(degenerate, (1.0, 1.0))


# -----------------------------------------------------------------------------
# Domain identifiers
# -----------------------------------------------------------------------------

_PATTERN = re.compile(r"^(insp|part|frame|trace)-\d{13}-[0-9a-f]{8}$")


def test_all_generators_match_format() -> None:
    for value in (
        ids.new_inspection_id(),
        ids.new_part_id(),
        ids.new_frame_id(),
        ids.new_trace_id(),
    ):
        assert _PATTERN.match(value), value


def test_prefixes_are_correct() -> None:
    assert ids.new_inspection_id().startswith("insp-")
    assert ids.new_part_id().startswith("part-")
    assert ids.new_frame_id().startswith("frame-")
    assert ids.new_trace_id().startswith("trace-")


def test_ids_are_unique() -> None:
    generated = {ids.new_inspection_id() for _ in range(1000)}
    assert len(generated) == 1000


def test_injected_clock_and_random_are_deterministic() -> None:
    value = ids.new_inspection_id(now_ns=1_700_000_000_000_000_000, rand_hex="deadbeef")
    assert value == "insp-1700000000000-deadbeef"


def test_ids_are_time_ordered_by_millisecond() -> None:
    earlier = ids.new_part_id(now_ns=1_000_000_000_000_000, rand_hex="ffffffff")
    later = ids.new_part_id(now_ns=2_000_000_000_000_000, rand_hex="00000000")
    assert earlier < later


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

def make_clock(times: list[float]) -> Callable[[], float]:
    """Return a clock that yields successive values from ``times``."""
    iterator = iter(times)
    return lambda: next(iterator)


def test_stopwatch_elapsed() -> None:
    clock = make_clock([100.0, 100.25])
    watch = timing.Stopwatch(clock=clock)
    assert watch.elapsed_s() == 0.25


def test_stopwatch_elapsed_ms() -> None:
    clock = make_clock([10.0, 10.5])
    watch = timing.Stopwatch(clock=clock)
    assert watch.elapsed_ms() == 500.0


def test_stopwatch_reset() -> None:
    clock = make_clock([1.0, 5.0, 6.0])
    watch = timing.Stopwatch(clock=clock)  # start = 1.0
    watch.reset()  # start = 5.0
    assert watch.elapsed_s() == 1.0  # 6.0 - 5.0


def test_deadline_not_expired_then_expired() -> None:
    clock = make_clock([0.0, 0.4, 1.1])
    deadline = timing.Deadline(1.0, clock=clock)  # deadline = 1.0
    assert deadline.expired() is False  # now 0.4
    assert deadline.expired() is True  # now 1.1


def test_deadline_remaining() -> None:
    clock = make_clock([0.0, 0.3])
    deadline = timing.Deadline(1.0, clock=clock)
    assert deadline.remaining_s() == 0.7


def test_deadline_from_ms() -> None:
    clock = make_clock([0.0, 0.25])
    deadline = timing.Deadline.from_ms(500.0, clock=clock)  # deadline = 0.5
    assert deadline.expired() is False


def test_measure_context_manager() -> None:
    clock = make_clock([2.0, 2.75])
    with timing.measure(clock=clock) as watch:
        pass
    assert watch.elapsed_ms() == 750.0


# -----------------------------------------------------------------------------
# Value types
# -----------------------------------------------------------------------------

def test_roi_center_and_validation() -> None:
    roi = types.ROI(label="pad", x=10.0, y=20.0, width=4.0, height=6.0)
    assert roi.center == (12.0, 23.0)


def test_roi_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        types.ROI(label="bad", x=0.0, y=0.0, width=-1.0, height=1.0)


def test_roi_roundtrip() -> None:
    roi = types.ROI(label="a", x=1.0, y=2.0, width=3.0, height=4.0, angle_deg=5.0)
    assert types.ROI.from_dict(roi.to_dict()) == roi


def test_roi_is_frozen() -> None:
    roi = types.ROI(label="a", x=0.0, y=0.0, width=1.0, height=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        roi.x = 9.0  # type: ignore[misc]


def test_pose_compose_inverse_transform() -> None:
    pose = types.Pose(3.0, -1.0, 90.0)
    identity = pose.compose(pose.inverse())
    assert identity.x == pytest.approx(0.0, abs=1e-9)
    assert identity.y == pytest.approx(0.0, abs=1e-9)
    assert identity.theta_deg == pytest.approx(0.0, abs=1e-9)
    tx, ty = pose.transform_point((1.0, 0.0))
    assert (tx, ty) == pytest.approx((3.0, 0.0), abs=1e-9)


def test_pose_roundtrip_and_as_tuple() -> None:
    pose = types.Pose(1.0, 2.0, 30.0)
    assert pose.as_tuple() == (1.0, 2.0, 30.0)
    assert types.Pose.from_dict(pose.to_dict()) == pose


def test_tolerance_validation() -> None:
    with pytest.raises(ValueError, match="minus"):
        types.Tolerance(minus=-1.0, plus=1.0)
    with pytest.raises(ValueError, match="plus"):
        types.Tolerance(minus=1.0, plus=-1.0)


def test_tolerance_roundtrip() -> None:
    tol = types.Tolerance(minus=0.1, plus=None)
    assert types.Tolerance.from_dict(tol.to_dict()) == tol


def test_measurement_spec_bounds_and_contains() -> None:
    spec = types.MeasurementSpec(
        name="width",
        nominal=10.0,
        tolerance=types.Tolerance(minus=0.2, plus=0.3),
        unit="mm",
    )
    assert spec.lower == pytest.approx(9.8)
    assert spec.upper == pytest.approx(10.3)
    assert spec.contains(9.8) is True
    assert spec.contains(10.3) is True
    assert spec.contains(9.79) is False
    assert spec.contains(10.31) is False


def test_measurement_spec_unbounded_sides() -> None:
    spec = types.MeasurementSpec(
        name="gap",
        nominal=5.0,
        tolerance=types.Tolerance(minus=None, plus=1.0),
        unit="mm",
    )
    assert spec.lower is None
    assert spec.contains(-100.0) is True
    assert spec.contains(6.0) is True
    assert spec.contains(6.1) is False


def test_measurement_spec_unbounded_above() -> None:
    spec = types.MeasurementSpec(
        name="clearance",
        nominal=5.0,
        tolerance=types.Tolerance(minus=1.0, plus=None),
        unit="mm",
    )
    assert spec.upper is None
    assert spec.contains(1_000.0) is True
    assert spec.contains(4.0) is True
    assert spec.contains(3.9) is False


def test_measurement_spec_roundtrip() -> None:
    spec = types.MeasurementSpec(
        name="d",
        nominal=1.0,
        tolerance=types.Tolerance(minus=0.1, plus=0.1),
        unit="mm",
    )
    assert types.MeasurementSpec.from_dict(spec.to_dict()) == spec


def test_measurement_roundtrip_with_and_without_spec() -> None:
    spec = types.MeasurementSpec(
        name="w",
        nominal=2.0,
        tolerance=types.Tolerance(minus=0.1, plus=0.1),
        unit="mm",
    )
    with_spec = types.Measurement(
        name="w", value=2.05, unit="mm", spec=spec, in_tolerance=True
    )
    without_spec = types.Measurement(name="raw", value=3.0, unit="px")
    assert types.Measurement.from_dict(with_spec.to_dict()) == with_spec
    assert types.Measurement.from_dict(without_spec.to_dict()) == without_spec


def test_frame_image_excluded_from_equality() -> None:
    from datetime import UTC, datetime

    ts = datetime(2026, 1, 1, tzinfo=UTC)
    a = types.RawFrame(
        image=object(),
        camera_id="cam0",
        frame_id="frame-1",
        timestamp_monotonic=1.0,
        timestamp_utc=ts,
    )
    b = types.RawFrame(
        image=object(),  # different object, but excluded from compare
        camera_id="cam0",
        frame_id="frame-1",
        timestamp_monotonic=1.0,
        timestamp_utc=ts,
    )
    assert a == b


def test_rectified_frame_construction() -> None:
    from datetime import UTC, datetime

    frame = types.RectifiedFrame(
        image=object(),
        camera_id="cam0",
        frame_id="frame-1",
        calibration_ver="calib-1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert frame.calibration_ver == "calib-1"
    assert frame.trigger_id is None


# -----------------------------------------------------------------------------
# Result shapes
# -----------------------------------------------------------------------------

def _defect() -> result.Defect:
    return result.Defect(
        defect_class=DefectClass.SCRATCH,
        severity=Severity.MAJOR,
        score=0.87,
        roi=types.ROI(label="r", x=0.0, y=0.0, width=1.0, height=1.0),
        description="hairline scratch",
    )


def _measurement() -> types.Measurement:
    return types.Measurement(name="width", value=10.1, unit="mm")


def _defect_measurement() -> result.DefectMeasurement:
    return result.DefectMeasurement(
        bbox=(1, 2, 3, 4), area_px2=12, area_um2=48.0, aspect_ratio=1.33, morphology="particle"
    )


def test_defect_roundtrip_full_and_minimal() -> None:
    full = _defect()
    minimal = result.Defect(defect_class=DefectClass.ANOMALY, severity=Severity.MINOR)
    assert result.Defect.from_dict(full.to_dict()) == full
    assert result.Defect.from_dict(minimal.to_dict()) == minimal


def test_metrology_result_roundtrip() -> None:
    partial = result.MetrologyResult(
        measurements=(_measurement(),), defects=(_defect(),)
    )
    assert result.MetrologyResult.from_dict(partial.to_dict()) == partial


def test_anomaly_result_roundtrip() -> None:
    partial = result.AnomalyResult(
        score=0.9,
        threshold=0.5,
        is_anomalous=True,
        heatmap_ref="img/heatmap-1.png",
        defects=(_defect(),),
    )
    assert result.AnomalyResult.from_dict(partial.to_dict()) == partial


def test_classical_result_roundtrip() -> None:
    partial = result.ClassicalResult(defects=(_defect(),))
    assert result.ClassicalResult.from_dict(partial.to_dict()) == partial


def test_partial_results_are_partial_result_subtypes() -> None:
    for cls in (result.MetrologyResult, result.AnomalyResult, result.ClassicalResult):
        assert issubclass(cls, result.PartialResult)


def test_partial_result_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        result.PartialResult()  # type: ignore[abstract]


def test_inspection_result_lossless_roundtrip() -> None:
    original = result.InspectionResult(
        inspection_id="insp-1",
        part_id="part-1",
        station_id="station-A",
        verdict=Verdict.FAIL,
        recipe_ver="recipe-3",
        model_ver="model-2",
        calib_ver="calib-1",
        cycle_time_ms=142.5,
        timestamp_utc=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        measurements=(_measurement(),),
        defects=(_defect(),),
        anomaly_score=0.91,
        image_refs=("img/raw-1.png", "img/overlay-1.png"),
        defect_measurements=(_defect_measurement(),),
        drift_status="NOMINAL",
    )
    restored = result.InspectionResult.from_dict(original.to_dict())
    assert restored == original


def test_inspection_result_json_serializable() -> None:
    import json

    original = result.InspectionResult(
        inspection_id="insp-2",
        part_id="part-2",
        station_id="station-A",
        verdict=Verdict.PASS,
        recipe_ver="r",
        model_ver="m",
        calib_ver="c",
        cycle_time_ms=100.0,
        timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC),
    )
    payload = json.dumps(original.to_dict())
    restored = result.InspectionResult.from_dict(json.loads(payload))
    assert restored == original


def test_inspection_result_is_frozen() -> None:
    res = result.InspectionResult(
        inspection_id="insp-3",
        part_id="part-3",
        station_id="s",
        verdict=Verdict.REVIEW,
        recipe_ver="r",
        model_ver="m",
        calib_ver="c",
        cycle_time_ms=1.0,
        timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.verdict = Verdict.PASS  # type: ignore[misc]


def _retrieval_match() -> result.RetrievalMatch:
    return result.RetrievalMatch(
        vector_id=7,
        distance=0.12,
        dataset="mvtec_ad",
        category="bottle",
        defect_type="crack",
        image_path="img/ref-7.png",
        metadata={"seed": 1},
    )


def test_retrieval_match_roundtrip_full_and_minimal() -> None:
    full = _retrieval_match()
    minimal = result.RetrievalMatch(
        vector_id=0, distance=0.0, dataset="d", category="c", defect_type="t"
    )
    assert result.RetrievalMatch.from_dict(full.to_dict()) == full
    assert result.RetrievalMatch.from_dict(minimal.to_dict()) == minimal


def test_inspection_evidence_roundtrip_with_heatmap_region_and_matches() -> None:
    full = result.InspectionEvidence(
        sample_id="insp-1",
        category="bottle",
        anomaly_score=0.93,
        severity=Severity.MAJOR,
        model_ver="patchcore-v1",
        retrieval_matches=(_retrieval_match(),),
        heatmap_region="upper-right",
    )
    assert result.InspectionEvidence.from_dict(full.to_dict()) == full


def test_inspection_evidence_roundtrip_minimal_without_heatmap_region() -> None:
    minimal = result.InspectionEvidence(
        sample_id="insp-2",
        category="bottle",
        anomaly_score=None,
        severity=Severity.INFO,
        model_ver="patchcore-v1",
    )
    restored = result.InspectionEvidence.from_dict(minimal.to_dict())
    assert restored == minimal
    assert restored.heatmap_region is None


def test_advisory_report_rejects_confidence_score_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence_score"):
        result.AdvisoryReport(
            defect_classification="x",
            severity=Severity.MINOR,
            confidence_score=1.5,
            root_cause_hypothesis="h",
        )


# -----------------------------------------------------------------------------
# Abstraction seams
# -----------------------------------------------------------------------------

ABSTRACT_INTERFACES = [
    interfaces.CameraDriver,
    interfaces.InferenceEngine,
    interfaces.AnomalyDetector,
    interfaces.Inspector,
    interfaces.PLCTransport,
    interfaces.MessagePublisher,
    interfaces.ResultRepository,
    interfaces.RecipeStore,
]


@pytest.mark.parametrize("interface", ABSTRACT_INTERFACES)
def test_interfaces_cannot_be_instantiated(interface: type) -> None:
    with pytest.raises(TypeError):
        interface()  # type: ignore[abstract, call-arg]


# --- Minimal fakes proving each contract is implementable --------------------


class FakeCamera(interfaces.CameraDriver):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def capture(self, trigger_id: str | None = None) -> RawFrame:
        return RawFrame(
            image=object(),
            camera_id="cam0",
            frame_id="frame-1",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
            trigger_id=trigger_id,
        )

    def is_healthy(self) -> bool:
        return True


class FakeEngine(interfaces.InferenceEngine):
    @property
    def model_version(self) -> str:
        return "fake-1"

    def load(self, model_id: str) -> None: ...
    def warmup(self) -> None: ...
    def infer(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(inputs)

    def unload(self) -> None: ...


class FakeDetector(interfaces.AnomalyDetector):
    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        return AnomalyResult(score=0.0, threshold=0.5, is_anomalous=False)


class FakeInspector(interfaces.Inspector[object, object]):
    def inspect(self, part: object, recipe: object) -> MetrologyResult:
        return MetrologyResult()


class FakePlc(interfaces.PLCTransport):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool:
        return True

    def read_coils(self, address: int, count: int) -> tuple[bool, ...]:
        return tuple(False for _ in range(count))

    def write_coil(self, address: int, value: bool) -> None: ...
    def read_registers(self, address: int, count: int) -> tuple[int, ...]:
        return tuple(0 for _ in range(count))

    def write_registers(self, address: int, values: Sequence[int]) -> None: ...


class FakePublisher(interfaces.MessagePublisher):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool:
        return True

    def publish(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None: ...


class FakeRepository(interfaces.ResultRepository):
    def save_result(self, result: InspectionResult) -> None: ...
    def get_result(self, inspection_id: str) -> InspectionResult | None:
        return None

    def list_results(
        self, *, limit: int = 100, offset: int = 0
    ) -> tuple[InspectionResult, ...]:
        return ()


class FakeRecipeStore(interfaces.RecipeStore[str]):
    def load(self, recipe_id: str) -> str:
        return recipe_id

    def save(self, recipe: str) -> None: ...
    def list_ids(self) -> tuple[str, ...]:
        return ()


def test_fakes_satisfy_contracts() -> None:
    assert FakeCamera().capture("t").trigger_id == "t"
    assert FakeCamera().is_healthy() is True
    assert FakeEngine().model_version == "fake-1"
    assert FakeEngine().infer({"x": 1}) == {"x": 1}
    FakeEngine().load("m")
    FakeEngine().warmup()
    FakeEngine().unload()
    frame = RectifiedFrame(
        image=object(),
        camera_id="c",
        frame_id="f",
        calibration_ver="cal",
        timestamp_monotonic=0.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert FakeDetector().detect(frame).is_anomalous is False
    assert isinstance(FakeInspector().inspect(object(), object()), MetrologyResult)


def test_fake_plc_contract() -> None:
    plc = FakePlc()
    plc.connect()
    assert plc.is_connected() is True
    assert plc.read_coils(0, 3) == (False, False, False)
    assert plc.read_registers(0, 2) == (0, 0)
    plc.write_coil(1, True)
    plc.write_registers(0, [1, 2])
    plc.disconnect()


def test_fake_publisher_and_repository_and_store() -> None:
    pub = FakePublisher()
    pub.connect()
    pub.publish("topic", {"k": "v"}, qos=1, retain=True)
    assert pub.is_connected() is True
    pub.disconnect()

    repo = FakeRepository()
    res = InspectionResult(
        inspection_id="i",
        part_id="p",
        station_id="s",
        verdict=Verdict.PASS,
        recipe_ver="r",
        model_ver="m",
        calib_ver="c",
        cycle_time_ms=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    repo.save_result(res)
    assert repo.get_result("i") is None
    assert repo.list_results() == ()

    store = FakeRecipeStore()
    store.save("r1")
    assert store.load("r1") == "r1"
    assert store.list_ids() == ()


def test_public_api_reexports_resolve() -> None:
    import adaptivevision.common as common

    for name in common.__all__:
        assert hasattr(common, name), name
