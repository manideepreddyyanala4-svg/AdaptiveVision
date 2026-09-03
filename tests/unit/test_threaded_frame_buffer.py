"""Unit tests for :class:`ThreadedFrameBuffer` (Milestone M21)."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest

from adaptivevision.camera import ThreadedFrameBuffer, build_frame
from adaptivevision.camera import NullCameraDriver
from adaptivevision.common import CameraKind
from adaptivevision.common import AcquisitionError
from adaptivevision.common import CameraDriver
from adaptivevision.common import RawFrame
from adaptivevision.config import CameraConfig

_WAIT_TIMEOUT_S = 2.0


def _wait_until(predicate: Callable[[], bool], *, timeout: float = _WAIT_TIMEOUT_S) -> bool:
    """Poll ``predicate`` until it's true or ``timeout`` elapses.

    Returns:
        Whether ``predicate`` became true before the timeout -- bounded, so a
        broken test fails fast instead of hanging the suite.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def _camera_config() -> CameraConfig:
    return CameraConfig(
        camera_id="cam0", kind=CameraKind.AREA_SCAN_2D, width=4, height=4, fps=30.0
    )


class _CountingDriver(CameraDriver):
    """Captures an incrementing pixel value each call, signaling an Event."""

    def __init__(self) -> None:
        self.count = 0
        self._opened = False
        self.captured = threading.Event()

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def is_healthy(self) -> bool:
        return self._opened

    def capture(self, trigger_id: str | None = None) -> RawFrame:
        self.count += 1
        # int32, not uint8: an unthrottled capture loop can run into the
        # thousands of iterations during a single test, and uint8 silently
        # wraps at 256 -- which looks exactly like ThreadedFrameBuffer
        # returning a stale frame when it's actually this fixture lying.
        frame = build_frame(np.full((2, 2), self.count, dtype=np.int32), "cam0")
        self.captured.set()
        return frame


class _FlakyDriver(CameraDriver):
    """Raises for the first ``fail_count`` captures, then succeeds."""

    #: Shared sentinel image so successful captures are distinguishable.
    _SUCCESS_IMAGE: ClassVar[np.ndarray] = np.ones((2, 2), dtype=np.uint8)

    def __init__(self, fail_count: int) -> None:
        self._remaining_failures = fail_count
        self._opened = False
        self.succeeded = threading.Event()

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def is_healthy(self) -> bool:
        return self._opened

    def capture(self, trigger_id: str | None = None) -> RawFrame:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            msg = "simulated capture failure"
            raise AcquisitionError(msg)
        self.succeeded.set()
        return build_frame(self._SUCCESS_IMAGE, "cam0")


class _FullThenEmptyQueue:
    """Simulates the narrow race in ``_push``: ``put_nowait`` reports the
    buffer full, but by the time ``_push`` tries to evict the oldest entry to
    make room, a concurrent reader has already drained it."""

    def __init__(self) -> None:
        self.put_calls = 0

    def put_nowait(self, item: object) -> None:
        self.put_calls += 1
        if self.put_calls == 1:
            raise queue.Full
        # Second attempt (after the simulated concurrent drain) succeeds.

    def get_nowait(self) -> object:
        raise queue.Empty


def test_push_recovers_when_a_concurrent_reader_wins_the_evict_race() -> None:
    driver = NullCameraDriver(_camera_config())
    driver.open()
    buffer = ThreadedFrameBuffer(driver)
    fake_queue = _FullThenEmptyQueue()
    buffer._queue = fake_queue  # type: ignore[assignment]

    buffer._push(build_frame(np.zeros((2, 2), dtype=np.uint8), "cam0"))

    assert fake_queue.put_calls == 2


def test_get_latest_frame_returns_none_before_start() -> None:
    buffer = ThreadedFrameBuffer(NullCameraDriver(_camera_config()))
    assert buffer.get_latest_frame() is None
    assert buffer.is_running is False


def test_start_stop_lifecycle_and_idempotency() -> None:
    driver = NullCameraDriver(_camera_config())
    driver.open()
    buffer = ThreadedFrameBuffer(driver)

    buffer.start()
    assert buffer.is_running is True
    buffer.start()  # idempotent, must not raise or spawn a second thread
    assert buffer.is_running is True

    buffer.stop()
    assert buffer.is_running is False
    buffer.stop()  # idempotent


def test_get_latest_frame_returns_captured_frame() -> None:
    driver = NullCameraDriver(_camera_config())
    driver.open()
    buffer = ThreadedFrameBuffer(driver)
    buffer.start()
    try:
        captured: list[RawFrame] = []

        def _try_capture() -> bool:
            frame = buffer.get_latest_frame()
            if frame is not None:
                captured.append(frame)
            return bool(captured)

        assert _wait_until(_try_capture)
        assert captured[0].image.shape == (4, 4)
    finally:
        buffer.stop()


def test_get_latest_frame_returns_most_recent_and_drains_backlog() -> None:
    driver = _CountingDriver()
    driver.open()
    buffer = ThreadedFrameBuffer(driver, maxsize=10)
    buffer.start()
    try:
        assert driver.captured.wait(timeout=_WAIT_TIMEOUT_S)
        # Let a handful more captures land so the buffer holds a backlog.
        assert _wait_until(lambda: driver.count >= 5)
        # Stop the producer *before* reading driver.count: it keeps
        # incrementing on the background thread otherwise, racing whatever
        # count we'd compare against next.
        buffer.stop()

        latest = buffer.get_latest_frame()
        assert latest is not None
        assert int(latest.image[0, 0]) == driver.count

        # The backlog was drained by the read above: nothing stale left queued.
        assert buffer.get_latest_frame() is None
    finally:
        buffer.stop()


def test_overflow_drops_oldest_and_never_blocks_capture_thread() -> None:
    driver = _CountingDriver()
    driver.open()
    buffer = ThreadedFrameBuffer(driver, maxsize=2)
    buffer.start()
    try:
        assert driver.captured.wait(timeout=_WAIT_TIMEOUT_S)
        assert _wait_until(lambda: driver.count >= 20)
        # The capture thread must still be alive after far outrunning maxsize.
        assert buffer.is_running is True
        buffer.stop()  # deterministic before comparing against driver.count

        latest = buffer.get_latest_frame()
        assert latest is not None
        assert int(latest.image[0, 0]) == driver.count
    finally:
        buffer.stop()


def test_capture_failure_is_recorded_and_loop_recovers() -> None:
    driver = _FlakyDriver(fail_count=5)
    driver.open()
    buffer = ThreadedFrameBuffer(driver)
    buffer.start()
    try:
        assert driver.succeeded.wait(timeout=_WAIT_TIMEOUT_S)
        error = buffer.last_error()
        assert error is not None
        assert isinstance(error, AcquisitionError)

        captured: list[RawFrame] = []

        def _try_capture() -> bool:
            frame = buffer.get_latest_frame()
            if frame is not None:
                captured.append(frame)
            return bool(captured)

        assert _wait_until(_try_capture)
    finally:
        buffer.stop()


def test_poll_interval_is_respected_without_breaking_capture() -> None:
    driver = NullCameraDriver(_camera_config())
    driver.open()
    buffer = ThreadedFrameBuffer(driver, poll_interval_s=0.01)
    buffer.start()
    try:
        captured: list[RawFrame] = []

        def _try_capture() -> bool:
            frame = buffer.get_latest_frame()
            if frame is not None:
                captured.append(frame)
            return bool(captured)

        assert _wait_until(_try_capture)
    finally:
        buffer.stop()
