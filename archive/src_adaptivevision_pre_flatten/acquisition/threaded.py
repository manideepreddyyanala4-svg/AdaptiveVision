"""Non-blocking, threaded camera frame acquisition (Milestone M21).

:class:`~adaptivevision.common.interfaces.CameraDriver` is deliberately a
blocking, single-threaded seam ("a single acquisition thread owns the
driver," per its own docstring) -- the right default for the walking
skeleton, but not for a line where a slow or stalled capture call must never
stall whatever consumes frames next (model inference). This module wraps any
driver, by composition rather than by changing the seam, in a
producer/consumer buffer: a background thread continuously calls
``driver.capture()`` and pushes into a bounded queue, while
:meth:`ThreadedFrameBuffer.get_latest_frame` always returns immediately with
the newest frame available.
"""

from __future__ import annotations

import queue
import threading
import time

from adaptivevision.common.interfaces import CameraDriver
from adaptivevision.common.types import RawFrame


class ThreadedFrameBuffer:
    """Runs a :class:`CameraDriver` on a dedicated background thread.

    Overflow policy: when the buffer is full, capturing a new frame discards
    the oldest buffered one to make room. The capture thread therefore never
    blocks waiting for a slow consumer, and :meth:`get_latest_frame` always
    hands back the most recent frame the camera has produced rather than
    working through a backlog of stale ones -- the right trade on a live
    line, where an old frame is not worth inspecting.

    A capture failure (any exception from ``driver.capture()``, not only
    :class:`~adaptivevision.common.errors.AcquisitionError`) is recorded via
    :meth:`last_error` and the loop keeps running -- a background thread that
    dies silently on the first unexpected error would be a worse failure mode
    than a slightly-broad catch here; Python does not propagate a thread's
    exception to its caller on its own.

    Args:
        driver: The camera driver to wrap. The caller owns its ``open()``/
            ``close()`` lifecycle (call ``open()`` before :meth:`start`),
            matching every other use of :class:`CameraDriver` in this
            codebase (see ``app.app.build_camera``).
        maxsize: Bounded buffer depth.
        poll_interval_s: Optional delay between capture attempts. ``None``
            (the default) captures as fast as the driver allows -- fine for
            a real camera whose ``capture()`` call itself takes real time,
            wasteful for a driver like
            :class:`~adaptivevision.acquisition.camera.NullCameraDriver` that
            returns instantly; pass e.g. ``1.0 / camera_config.fps`` to pace
            it explicitly.
    """

    def __init__(
        self,
        driver: CameraDriver,
        *,
        maxsize: int = 10,
        poll_interval_s: float | None = None,
    ) -> None:
        """Initialize the buffer without starting the capture thread."""
        self._driver = driver
        self._poll_interval_s = poll_interval_s
        self._queue: queue.Queue[RawFrame] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error_lock = threading.Lock()
        self._last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        """Whether the background capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background capture thread. A no-op if already running."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="ThreadedFrameBuffer", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float | None = 2.0) -> None:
        """Signal the capture thread to stop and wait for it to exit.

        Args:
            timeout: Maximum time to wait for the thread to exit.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def get_latest_frame(self) -> RawFrame | None:
        """Return the most recently captured frame without blocking.

        Returns:
            The newest buffered frame, or ``None`` if none has been captured
            yet (or none survived the overflow policy since the last read).
        """
        latest: RawFrame | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                return latest

    def last_error(self) -> Exception | None:
        """Return the most recent capture failure, without clearing it."""
        with self._error_lock:
            return self._last_error

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._driver.capture()
            except Exception as exc:  # noqa: BLE001 -- see class docstring
                with self._error_lock:
                    self._last_error = exc
            else:
                self._push(frame)
            if self._poll_interval_s is not None:
                time.sleep(self._poll_interval_s)

    def _push(self, frame: RawFrame) -> None:
        """Enqueue ``frame``, dropping the oldest buffered frame if full."""
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass  # a concurrent reader already drained it; retry the put
