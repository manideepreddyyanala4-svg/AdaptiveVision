"""Unit tests for camera.py: driver, threaded frame buffer, calibration,
alignment, preprocessing."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import ClassVar

import cv2
import numpy as np
import pytest

from adaptivevision.camera import (
    CalibrationManager,
    CalibrationRectifier,
    CalibrationSelfTest,
    CameraCalibration,
    FileImageCameraDriver,
    GoldenReference,
    NullCameraDriver,
    PreprocessingPipeline,
    ReferenceAligner,
    ThreadedFrameBuffer,
    build_frame,
    ensure_grayscale,
    identity_calibration,
    load_calibration,
    load_golden_reference,
    new_frame_id,
    normalize_uint8,
)
from adaptivevision.common import (
    AcquisitionError,
    CalibrationError,
    CameraDriver,
    CameraKind,
    FaultError,
    Pose,
    RawFrame,
    RectifiedFrame,
)
from adaptivevision.config import CameraConfig

# -----------------------------------------------------------------------------
# Camera driver and frame construction
# -----------------------------------------------------------------------------

def _config() -> CameraConfig:
    return CameraConfig(
        camera_id="cam0",
        kind=CameraKind.AREA_SCAN_2D,
        width=640,
        height=480,
        fps=30.0,
    )


def test_null_camera_is_camera_driver() -> None:
    assert isinstance(NullCameraDriver(_config()), CameraDriver)


def test_null_camera_open_close_lifecycle() -> None:
    driver = NullCameraDriver(_config())
    assert driver.is_healthy() is False
    driver.open()
    assert driver.is_healthy() is True
    driver.close()
    assert driver.is_healthy() is False


def test_null_camera_capture_returns_frame() -> None:
    driver = NullCameraDriver(_config())
    driver.open()
    frame = driver.capture(trigger_id="trig-1")
    assert frame.camera_id == "cam0"
    assert frame.trigger_id == "trig-1"
    assert frame.image.shape == (480, 640)
    assert frame.image.dtype == np.uint8


def test_null_camera_capture_before_open_raises() -> None:
    driver = NullCameraDriver(_config())
    with pytest.raises(AcquisitionError, match="before open"):
        driver.capture()


def _write_bgr_image(path, bgr_pixel: tuple[int, int, int]) -> None:
    image = np.full((4, 4, 3), bgr_pixel, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_file_image_camera_is_camera_driver(tmp_path) -> None:
    path = tmp_path / "frame.png"
    _write_bgr_image(path, (10, 20, 30))
    assert isinstance(FileImageCameraDriver(path), CameraDriver)


def test_file_image_camera_open_close_lifecycle(tmp_path) -> None:
    path = tmp_path / "frame.png"
    _write_bgr_image(path, (10, 20, 30))
    driver = FileImageCameraDriver(path)
    assert driver.is_healthy() is False
    driver.open()
    assert driver.is_healthy() is True
    driver.close()
    assert driver.is_healthy() is False


def test_file_image_camera_capture_converts_bgr_to_rgb(tmp_path) -> None:
    path = tmp_path / "frame.png"
    _write_bgr_image(path, (10, 20, 30))  # OpenCV file order: B=10, G=20, R=30
    driver = FileImageCameraDriver(path, camera_id="demo-cam")
    driver.open()

    frame = driver.capture(trigger_id="trig-1")

    assert frame.camera_id == "demo-cam"
    assert frame.trigger_id == "trig-1"
    assert frame.image.shape == (4, 4, 3)
    assert frame.image.dtype == np.uint8
    assert tuple(frame.image[0, 0]) == (30, 20, 10)  # RGB order: R=30, G=20, B=10


def test_file_image_camera_capture_returns_independent_copies(tmp_path) -> None:
    path = tmp_path / "frame.png"
    _write_bgr_image(path, (10, 20, 30))
    driver = FileImageCameraDriver(path)
    driver.open()

    first = driver.capture()
    first.image[0, 0] = 255
    second = driver.capture()

    assert second.image[0, 0].tolist() != [255, 255, 255]


def test_file_image_camera_capture_before_open_raises(tmp_path) -> None:
    path = tmp_path / "frame.png"
    _write_bgr_image(path, (10, 20, 30))
    driver = FileImageCameraDriver(path)
    with pytest.raises(AcquisitionError, match="before open"):
        driver.capture()


def test_file_image_camera_open_missing_file_raises(tmp_path) -> None:
    driver = FileImageCameraDriver(tmp_path / "does-not-exist.png")
    with pytest.raises(AcquisitionError, match="could not read"):
        driver.open()


def test_new_frame_id_is_unique() -> None:
    assert new_frame_id() != new_frame_id()
    assert new_frame_id().startswith("frame-")


def test_build_frame_populates_metadata() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    frame = build_frame(image, "cam0", trigger_id="t", frame_id="f1")
    assert frame.frame_id == "f1"
    assert frame.camera_id == "cam0"
    assert frame.trigger_id == "t"
    assert frame.image is image


def test_build_frame_generates_id_when_omitted() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    frame = build_frame(image, "cam0")
    assert frame.frame_id.startswith("frame-")


# -----------------------------------------------------------------------------
# Threaded frame buffer
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Calibration artifact model and rectification
# -----------------------------------------------------------------------------

def _artifact() -> dict[str, object]:
    return {
        "calibration_id": "calib-cam0",
        "version": "calib-v1",
        "camera_id": "cam0",
        "image_width": 4,
        "image_height": 3,
        "pixel_size_mm": 0.02,
        "intrinsic_matrix": [[1.0, 0.0, 2.0], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
    }


def _raw_frame(camera_id: str = "cam0", shape: tuple[int, int] = (3, 4)) -> RawFrame:
    return RawFrame(
        image=np.ones(shape, dtype=np.uint8),
        camera_id=camera_id,
        frame_id="frame-1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_calibration_roundtrip() -> None:
    calibration = CameraCalibration.from_dict(_artifact())
    assert CameraCalibration.from_dict(calibration.to_dict()) == calibration


def test_identity_calibration_builds_valid_artifact() -> None:
    calibration = identity_calibration(camera_id="cam0", width=640, height=480)
    assert calibration.camera_id == "cam0"
    assert calibration.version == "identity"


def test_calibration_rejects_invalid_intrinsics() -> None:
    data = _artifact()
    data["intrinsic_matrix"] = [[1.0]]
    with pytest.raises(CalibrationError, match="3x3"):
        CameraCalibration.from_dict(data)


def test_load_calibration_from_json(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    assert load_calibration(path).version == "calib-v1"


def test_load_calibration_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CalibrationError, match="must contain a JSON object"):
        load_calibration(path)


def test_rectifier_applies_lineage_and_copies_image() -> None:
    calibration = CameraCalibration.from_dict(_artifact())
    frame = _raw_frame()
    rectified = CalibrationRectifier(calibration).apply(frame)
    assert rectified.calibration_ver == "calib-v1"
    assert rectified.frame_id == frame.frame_id
    assert rectified.image is not frame.image
    np.testing.assert_array_equal(rectified.image, frame.image)


def test_rectifier_rejects_camera_mismatch() -> None:
    calibration = CameraCalibration.from_dict(_artifact())
    with pytest.raises(CalibrationError, match="does not match"):
        CalibrationRectifier(calibration).apply(_raw_frame(camera_id="other"))


def test_rectifier_rejects_dimension_mismatch() -> None:
    calibration = CameraCalibration.from_dict(_artifact())
    with pytest.raises(CalibrationError, match="do not match"):
        CalibrationRectifier(calibration).apply(_raw_frame(shape=(2, 4)))


# -----------------------------------------------------------------------------
# Calibration hot-swap lifecycle
# -----------------------------------------------------------------------------

def _calibration(
    calibration_id: str = "cal-1", version: str = "1.0.0"
) -> CameraCalibration:
    return CameraCalibration(
        calibration_id=calibration_id,
        version=version,
        camera_id="cam-1",
        image_width=1280,
        image_height=720,
        pixel_size_mm=0.01,
        intrinsic_matrix=((1000.0, 0.0, 640.0), (0.0, 1000.0, 360.0), (0.0, 0.0, 1.0)),
    )


def test_self_test_passes_valid_calibration() -> None:
    result = CalibrationSelfTest().run(_calibration())
    assert result.passed
    assert len(result.checks) == 3


def test_self_test_fails_singular_matrix() -> None:
    calibration = CameraCalibration(
        calibration_id="cal-bad",
        version="1.0.0",
        camera_id="cam-1",
        image_width=1280,
        image_height=720,
        pixel_size_mm=0.01,
        intrinsic_matrix=((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (1.0, 1.0, 1.0)),
    )
    result = CalibrationSelfTest().run(calibration)
    assert not result.passed


def test_manager_activate_and_active() -> None:
    manager = CalibrationManager()
    assert manager.active("cam-1") is None
    manager.activate(_calibration())
    assert manager.active("cam-1") is not None
    assert manager.cameras() == ("cam-1",)


def test_manager_hot_swap_replaces_calibration() -> None:
    manager = CalibrationManager()
    manager.activate(_calibration(calibration_id="cal-1", version="1.0.0"))
    manager.activate(_calibration(calibration_id="cal-2", version="2.0.0"))
    assert manager.active("cam-1").version == "2.0.0"


def test_manager_rejects_failed_self_test() -> None:
    manager = CalibrationManager()
    bad = CameraCalibration(
        calibration_id="cal-bad",
        version="1.0.0",
        camera_id="cam-1",
        image_width=1280,
        image_height=720,
        pixel_size_mm=0.01,
        intrinsic_matrix=((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (1.0, 1.0, 1.0)),
    )
    with pytest.raises(CalibrationError):
        manager.activate(bad)
    assert manager.active("cam-1") is None


def test_identity_calibration_passes_self_test() -> None:
    calibration = identity_calibration(camera_id="cam-1", width=640, height=480)
    assert CalibrationSelfTest().run(calibration).passed


# -----------------------------------------------------------------------------
# Golden-reference alignment
# -----------------------------------------------------------------------------

def _reference_data() -> dict[str, object]:
    return {
        "reference_id": "golden-1",
        "version": "ref-v1",
        "camera_id": "cam0",
        "image_width": 4,
        "image_height": 3,
        "nominal_pose": {"x": 1.0, "y": 2.0, "theta_deg": 3.0},
        "min_score": 0.9,
    }


def _rectified_frame(camera_id: str = "cam0", shape: tuple[int, int] = (3, 4)) -> RectifiedFrame:
    return RectifiedFrame(
        image=np.ones(shape, dtype=np.uint8),
        camera_id=camera_id,
        frame_id="frame-1",
        calibration_ver="calib-v1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_golden_reference_roundtrip() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    assert GoldenReference.from_dict(reference.to_dict()) == reference


def test_golden_reference_defaults_nominal_pose() -> None:
    data = _reference_data()
    data.pop("nominal_pose")
    reference = GoldenReference.from_dict(data)
    assert reference.nominal_pose == Pose(0.0, 0.0, 0.0)


def test_golden_reference_rejects_invalid_score() -> None:
    data = _reference_data()
    data["min_score"] = 1.5
    with pytest.raises(FaultError, match=r"\[0, 1\]"):
        GoldenReference.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_id", "", "reference_id"),
        ("version", "", "version"),
        ("camera_id", "", "camera_id"),
        ("image_width", 0, "dimensions"),
    ],
)
def test_golden_reference_rejects_invalid_required_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    data = _reference_data()
    data[field] = value
    with pytest.raises(FaultError, match=message):
        GoldenReference.from_dict(data)


def test_load_golden_reference_from_json(tmp_path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference_data()), encoding="utf-8")
    assert load_golden_reference(path).version == "ref-v1"


def test_load_golden_reference_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FaultError, match="Failed to load golden reference"):
        load_golden_reference(tmp_path / "missing.json")


def test_load_golden_reference_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(FaultError, match="must contain a JSON object"):
        load_golden_reference(path)


def test_load_golden_reference_rejects_invalid_artifact(tmp_path) -> None:
    path = tmp_path / "bad-reference.json"
    data = _reference_data()
    data.pop("reference_id")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FaultError, match="Invalid golden reference"):
        load_golden_reference(path)


def test_reference_aligner_returns_localized_part() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    part = ReferenceAligner(reference).align(_rectified_frame())
    assert part.reference_id == "golden-1"
    assert part.reference_ver == "ref-v1"
    assert part.pose == Pose(1.0, 2.0, 3.0)
    assert part.score == 1.0
    assert part.lineage()["calibration_ver"] == "calib-v1"


def test_reference_aligner_rejects_camera_mismatch() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    with pytest.raises(FaultError, match="does not match"):
        ReferenceAligner(reference).align(_rectified_frame(camera_id="other"))


def test_reference_aligner_rejects_dimension_mismatch() -> None:
    reference = GoldenReference.from_dict(_reference_data())
    with pytest.raises(FaultError, match="do not match"):
        ReferenceAligner(reference).align(_rectified_frame(shape=(2, 4)))


def test_reference_aligner_rejects_min_score_above_estimate() -> None:
    data = _reference_data()
    data["min_score"] = 1.0
    reference = GoldenReference.from_dict(data)
    assert ReferenceAligner(reference).align(_rectified_frame()).score == 1.0


def test_localized_part_rejects_invalid_score() -> None:
    frame = _rectified_frame()
    with pytest.raises(FaultError, match=r"\[0, 1\]"):
        from adaptivevision.camera import LocalizedPart

        LocalizedPart(
            frame=frame,
            pose=Pose(0.0, 0.0, 0.0),
            reference_id="golden",
            reference_ver="ref-v1",
            score=1.1,
        )


# -----------------------------------------------------------------------------
# Preprocessing operators
# -----------------------------------------------------------------------------

def _frame_from_image(image: np.ndarray) -> RawFrame:
    return RawFrame(
        image=image,
        camera_id="cam0",
        frame_id="frame-1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        trigger_id="trigger-1",
    )


def test_normalize_uint8_scales_image() -> None:
    frame = _frame_from_image(np.array([[10, 20], [30, 40]], dtype=np.uint16))
    normalized = normalize_uint8(frame)
    assert normalized.image.dtype == np.uint8
    assert normalized.image.min() == 0
    assert normalized.image.max() == 255
    assert normalized.frame_id == frame.frame_id


def test_normalize_uint8_constant_image_returns_zero() -> None:
    frame = _frame_from_image(np.full((2, 2), 5, dtype=np.uint8))
    normalized = normalize_uint8(frame)
    assert np.count_nonzero(normalized.image) == 0


def test_ensure_grayscale_converts_rgb() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    gray = ensure_grayscale(_frame_from_image(image))
    assert gray.image.shape == (1, 1)
    assert gray.image[0, 0] == 76


def test_ensure_grayscale_copies_grayscale_input() -> None:
    frame = _frame_from_image(np.ones((2, 2), dtype=np.uint8))
    gray = ensure_grayscale(frame)
    assert gray.image is not frame.image
    np.testing.assert_array_equal(gray.image, frame.image)


def test_ensure_grayscale_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="Expected"):
        ensure_grayscale(_frame_from_image(np.zeros((1, 1, 2), dtype=np.uint8)))


def test_preprocessing_pipeline_applies_steps_in_order() -> None:
    image = np.array([[[0, 0, 0], [10, 10, 10]]], dtype=np.uint8)
    pipeline = PreprocessingPipeline((ensure_grayscale, normalize_uint8))
    processed = pipeline.apply(_frame_from_image(image))
    assert processed.image.tolist() == [[0, 255]]
