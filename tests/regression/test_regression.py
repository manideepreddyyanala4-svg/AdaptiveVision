"""Regression suite (Milestone M17).

These tests exercise the full walking-skeleton pipeline end-to-end to guard
against regressions across milestones. They are intentionally lightweight and
dependency-free so they run in CI.
"""

from __future__ import annotations

from datetime import UTC, datetime

from adaptivevision.camera import NullCameraDriver
from adaptivevision.common import CameraKind, Verdict
from adaptivevision.common import InspectionResult
from adaptivevision.config import CameraConfig
from adaptivevision.orchestration import (
    FailureHandler,
    InspectionPipeline,
    ResultBuffer,
)


def _make_pipeline() -> InspectionPipeline:
    camera = NullCameraDriver(
        CameraConfig("cam0", CameraKind.AREA_SCAN_2D, 640, 480, 30.0)
    )
    camera.open()
    return InspectionPipeline(camera, station_id="station-1", recipe_ver="1.0.0")


def test_pipeline_runs_end_to_end() -> None:
    pipeline = _make_pipeline()
    result = pipeline.run(part_id="part-1")
    assert isinstance(result, InspectionResult)
    assert result.part_id == "part-1"
    assert result.verdict in (Verdict.PASS, Verdict.FAIL)
    assert result.cycle_time_ms >= 0.0


def test_pipeline_produces_unique_inspection_ids() -> None:

    pipeline = _make_pipeline()
    ids = {pipeline.run(part_id="part-1").inspection_id for _ in range(5)}
    assert len(ids) == 5


def test_failure_handler_buffers_and_flushes() -> None:
    persisted: list[str] = []
    attempts = 0

    def persist(result: InspectionResult) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("transient")
        persisted.append(result.inspection_id)

    buffer = ResultBuffer()
    handler = FailureHandler(persist, max_attempts=1, buffer=buffer)
    result = InspectionResult(
        inspection_id="insp-1",
        part_id="part-1",
        station_id="station-1",
        verdict=Verdict.PASS,
        recipe_ver="1.0.0",
        model_ver="1.0.0",
        calib_ver="1.0.0",
        cycle_time_ms=10.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    handler.handle(result)
    assert len(buffer) == 1
    remaining = handler.flush()
    assert remaining == ()
    assert persisted == ["insp-1"]
