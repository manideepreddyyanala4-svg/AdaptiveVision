"""Unit tests for orchestration.py: state machine, pipeline, scheduler, resilience."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from adaptivevision.app import StationController, build_camera, build_station
from adaptivevision.camera import LocalizedPart, NullCameraDriver
from adaptivevision.common import (
    CameraKind,
    FaultError,
    InspectionResult,
    MeasurementSpec,
    Pose,
    RawFrame,
    RectifiedFrame,
    StationState,
    Tolerance,
    Verdict,
)
from adaptivevision.config import CameraConfig, Recipe, StationConfig
from adaptivevision.metrology import (
    MetrologyInspector,
    StaticMeasurementSource,
)
from adaptivevision.orchestration import (
    CycleWatchdog,
    FailureHandler,
    InspectionPipeline,
    InspectionScheduler,
    ResultBuffer,
    StationStateMachine,
    new_inspection_id,
)

# -----------------------------------------------------------------------------
# State machine, pipeline, scheduler, watchdog
# -----------------------------------------------------------------------------

def test_state_machine_boot_path() -> None:
    sm = StationStateMachine()
    assert sm.state is StationState.INIT
    sm.transition(StationState.SELF_TEST)
    sm.transition(StationState.IDLE)
    assert sm.state is StationState.IDLE


def test_state_machine_invalid_transition_raises() -> None:
    sm = StationStateMachine()
    with pytest.raises(FaultError, match="Invalid station transition"):
        sm.transition(StationState.RUNNING)


def test_state_machine_can_transition() -> None:
    sm = StationStateMachine()
    assert sm.can_transition(StationState.SELF_TEST) is True
    assert sm.can_transition(StationState.RUNNING) is False


def test_state_machine_to_fault() -> None:
    sm = StationStateMachine()
    sm.transition(StationState.SELF_TEST)
    assert sm.to_fault() is StationState.FAULT


def test_state_machine_to_fault_noop_when_not_allowed() -> None:
    sm = StationStateMachine(initial=StationState.SHUTDOWN)
    assert sm.to_fault() is StationState.SHUTDOWN


# --- Pipeline ----------------------------------------------------------------


def _camera() -> NullCameraDriver:
    driver = NullCameraDriver(
        CameraConfig("cam0", CameraKind.AREA_SCAN_2D, 640, 480, 30.0)
    )
    driver.open()
    return driver


def test_pipeline_run_produces_result() -> None:
    pipeline = InspectionPipeline(_camera(), station_id="s1", recipe_ver="1.0")
    result = pipeline.run("part-1", trigger_id="t1")
    assert result.part_id == "part-1"
    assert result.station_id == "s1"
    assert result.recipe_ver == "1.0"
    assert result.verdict is Verdict.PASS
    assert result.cycle_time_ms >= 0.0
    assert len(result.image_refs) == 1


def test_pipeline_applies_preprocessing_and_rectification() -> None:
    seen: list[str] = []

    def preprocess(frame: RawFrame) -> RawFrame:
        seen.append("preprocess")
        return frame

    def rectify(frame: RawFrame) -> RectifiedFrame:
        seen.append("rectify")
        return RectifiedFrame(
            image=frame.image,
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            calibration_ver="calib-v1",
            timestamp_monotonic=frame.timestamp_monotonic,
            timestamp_utc=frame.timestamp_utc,
            trigger_id=frame.trigger_id,
        )

    pipeline = InspectionPipeline(
        _camera(),
        station_id="s1",
        recipe_ver="1.0",
        preprocessor=preprocess,
        rectifier=rectify,
    )
    result = pipeline.run("part-1")
    assert seen == ["preprocess", "rectify"]
    assert result.calib_ver == "calib-v1"


def test_pipeline_applies_alignment_after_rectification() -> None:
    seen: list[str] = []

    def rectify(frame: RawFrame) -> RectifiedFrame:
        seen.append("rectify")
        return RectifiedFrame(
            image=frame.image,
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            calibration_ver="calib-v1",
            timestamp_monotonic=frame.timestamp_monotonic,
            timestamp_utc=frame.timestamp_utc,
            trigger_id=frame.trigger_id,
        )

    def align(frame: RectifiedFrame) -> LocalizedPart:
        seen.append("align")
        return LocalizedPart(
            frame=frame,
            pose=frame_pose(),
            reference_id="golden",
            reference_ver="ref-v1",
            score=1.0,
        )

    pipeline = InspectionPipeline(
        _camera(),
        station_id="s1",
        recipe_ver="1.0",
        rectifier=rectify,
        aligner=align,
    )
    result = pipeline.run("part-1")
    assert seen == ["rectify", "align"]
    assert result.calib_ver == "calib-v1"


def frame_pose():
    """Return a nominal test pose."""
    return Pose(0.0, 0.0, 0.0)


def _aligned_part(frame: RectifiedFrame) -> LocalizedPart:
    return LocalizedPart(
        frame=frame,
        pose=frame_pose(),
        reference_id="golden",
        reference_ver="ref-v1",
        score=1.0,
    )


def _metrology_recipe() -> Recipe:
    return Recipe(
        recipe_id="recipe-1",
        version="v1",
        measurement_specs=(
            MeasurementSpec(
                name="width",
                nominal=10.0,
                tolerance=Tolerance(minus=0.2, plus=0.2),
                unit="mm",
            ),
        ),
    )


def test_pipeline_includes_metrology_measurements() -> None:
    def align(frame: RectifiedFrame) -> LocalizedPart:
        return _aligned_part(frame)

    inspector = MetrologyInspector(StaticMeasurementSource({"width": 10.1}).measure)
    pipeline = InspectionPipeline(
        _camera(),
        station_id="s1",
        recipe_ver="v1",
        aligner=align,
        recipe=_metrology_recipe(),
        metrology_inspector=inspector,
    )
    result = pipeline.run("part-1")
    assert result.verdict is Verdict.PASS
    assert result.measurements[0].name == "width"
    assert result.defects == ()


def test_pipeline_fails_on_metrology_defects() -> None:
    def align(frame: RectifiedFrame) -> LocalizedPart:
        return _aligned_part(frame)

    inspector = MetrologyInspector(StaticMeasurementSource({"width": 11.0}).measure)
    pipeline = InspectionPipeline(
        _camera(),
        station_id="s1",
        recipe_ver="v1",
        aligner=align,
        recipe=_metrology_recipe(),
        metrology_inspector=inspector,
    )
    result = pipeline.run("part-1")
    assert result.verdict is Verdict.FAIL
    assert len(result.defects) == 1


def test_new_inspection_id_unique() -> None:
    assert new_inspection_id() != new_inspection_id()
    assert new_inspection_id().startswith("inspection-")


# --- Scheduler ---------------------------------------------------------------


def test_scheduler_runs_cycles() -> None:
    scheduler = InspectionScheduler(
        InspectionPipeline(_camera(), station_id="s", recipe_ver="1")
    )
    results = scheduler.run_cycles(["p1", "p2", "p3"])
    assert [r.part_id for r in results] == ["p1", "p2", "p3"]


def test_scheduler_invokes_callback() -> None:
    scheduler = InspectionScheduler(
        InspectionPipeline(_camera(), station_id="s", recipe_ver="1")
    )
    seen: list[str] = []
    scheduler.run_cycles(["p1", "p2"], on_result=lambda r: seen.append(r.part_id))
    assert seen == ["p1", "p2"]


# --- Watchdog ----------------------------------------------------------------


def _result_with_cycle_time(cycle_time_ms: float) -> InspectionResult:
    return InspectionResult(
        inspection_id="i1",
        part_id="p",
        station_id="s",
        verdict=Verdict.PASS,
        recipe_ver="1",
        model_ver="",
        calib_ver="",
        cycle_time_ms=cycle_time_ms,
        timestamp_utc=datetime.now(UTC),
    )


def test_watchdog_no_violation() -> None:
    watchdog = CycleWatchdog(timeout_ms=1000.0)
    assert watchdog.check(_result_with_cycle_time(cycle_time_ms=500.0)) is False
    assert watchdog.violations == 0


def test_watchdog_violation() -> None:
    watchdog = CycleWatchdog(timeout_ms=100.0)
    assert watchdog.check(_result_with_cycle_time(cycle_time_ms=500.0)) is True
    assert watchdog.violations == 1


def test_watchdog_timeout_property() -> None:
    assert CycleWatchdog(timeout_ms=42.0).timeout_ms == 42.0


# --- Composition root --------------------------------------------------------


def test_build_camera_returns_null_driver_when_no_cameras() -> None:
    config = StationConfig(station_id="s", log_level="INFO")
    camera = build_camera(config)
    assert isinstance(camera, NullCameraDriver)


def test_build_camera_uses_configured_camera() -> None:
    config = StationConfig(
        station_id="s",
        log_level="INFO",
        cameras={"cam0": CameraConfig("cam0", CameraKind.AREA_SCAN_2D, 640, 480, 30.0)},
    )
    camera = build_camera(config)
    assert isinstance(camera, NullCameraDriver)


def test_build_station_boots_and_runs() -> None:
    config = StationConfig(station_id="s1", log_level="INFO")
    station = build_station(config)
    assert isinstance(station, StationController)
    assert station.state is StationState.INIT

    station.boot()
    assert station.state is StationState.IDLE

    station.ready()
    assert station.state is StationState.READY

    results = station.run(["part-1"])
    assert len(results) == 1
    assert results[0].verdict is Verdict.PASS
    assert station.state is StationState.READY

    station.shutdown()
    assert station.state is StationState.SHUTDOWN


def test_station_run_requires_ready_state() -> None:
    station = build_station(StationConfig(station_id="s", log_level="INFO"))
    with pytest.raises(FaultError):
        station.run(["part-1"])


# -----------------------------------------------------------------------------
# Result buffering and failure handling
# -----------------------------------------------------------------------------

def _result_with_id(inspection_id: str) -> InspectionResult:
    return InspectionResult(
        inspection_id=inspection_id,
        part_id="part-1",
        station_id="station-1",
        verdict=Verdict.PASS,
        recipe_ver="1.0.0",
        model_ver="1.0.0",
        calib_ver="1.0.0",
        cycle_time_ms=10.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_buffer_push_and_drain() -> None:
    buffer = ResultBuffer()
    buffer.push(_result_with_id("insp-1"))
    buffer.push(_result_with_id("insp-2"))
    assert len(buffer) == 2
    drained = buffer.drain()
    assert [r.inspection_id for r in drained] == ["insp-1", "insp-2"]
    assert len(buffer) == 0


def test_buffer_drops_oldest_when_full() -> None:
    buffer = ResultBuffer(capacity=2)
    buffer.push(_result_with_id("insp-1"))
    buffer.push(_result_with_id("insp-2"))
    buffer.push(_result_with_id("insp-3"))
    assert buffer.is_full()
    drained = buffer.drain()
    assert [r.inspection_id for r in drained] == ["insp-2", "insp-3"]


def test_buffer_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError):
        ResultBuffer(capacity=0)


def test_failure_handler_persists_on_first_attempt() -> None:
    persisted: list[str] = []

    def persist(result: InspectionResult) -> None:
        persisted.append(result.inspection_id)

    handler = FailureHandler(persist)
    outcome = handler.handle(_result_with_id("insp-1"))
    assert outcome.persisted
    assert outcome.attempts == 1
    assert not outcome.buffered
    assert persisted == ["insp-1"]


def test_failure_handler_retries_then_buffers() -> None:
    attempts = 0

    def persist(result: InspectionResult) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("persistence down")

    handler = FailureHandler(persist, max_attempts=3)
    outcome = handler.handle(_result_with_id("insp-1"))
    assert not outcome.persisted
    assert outcome.attempts == 3
    assert outcome.buffered
    assert len(handler._buffer) == 1  # type: ignore[attr-defined]


def test_failure_handler_succeeds_on_retry() -> None:
    attempts = 0

    def persist(result: InspectionResult) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("transient")

    handler = FailureHandler(persist, max_attempts=3)
    outcome = handler.handle(_result_with_id("insp-1"))
    assert outcome.persisted
    assert outcome.attempts == 2


def test_failure_handler_flush_retries_buffered() -> None:
    attempts = 0

    def persist(result: InspectionResult) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("transient")

    handler = FailureHandler(persist, max_attempts=1)
    handler.handle(_result_with_id("insp-1"))
    assert len(handler._buffer) == 1  # type: ignore[attr-defined]
    remaining = handler.flush()
    assert remaining == ()
    assert len(handler._buffer) == 0  # type: ignore[attr-defined]
