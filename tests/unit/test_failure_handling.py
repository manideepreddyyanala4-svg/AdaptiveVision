"""Unit tests for the M17 failure handling and result buffering."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from adaptivevision.common import Verdict
from adaptivevision.common import InspectionResult
from adaptivevision.orchestration import FailureHandler, ResultBuffer


def _result(inspection_id: str) -> InspectionResult:
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
    buffer.push(_result("insp-1"))
    buffer.push(_result("insp-2"))
    assert len(buffer) == 2
    drained = buffer.drain()
    assert [r.inspection_id for r in drained] == ["insp-1", "insp-2"]
    assert len(buffer) == 0


def test_buffer_drops_oldest_when_full() -> None:
    buffer = ResultBuffer(capacity=2)
    buffer.push(_result("insp-1"))
    buffer.push(_result("insp-2"))
    buffer.push(_result("insp-3"))
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
    outcome = handler.handle(_result("insp-1"))
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
    outcome = handler.handle(_result("insp-1"))
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
    outcome = handler.handle(_result("insp-1"))
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
    handler.handle(_result("insp-1"))
    assert len(handler._buffer) == 1  # type: ignore[attr-defined]
    remaining = handler.flush()
    assert remaining == ()
    assert len(handler._buffer) == 0  # type: ignore[attr-defined]
