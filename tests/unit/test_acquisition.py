"""Unit tests for :mod:`adaptivevision.acquisition`."""

from __future__ import annotations

import numpy as np
import pytest

from adaptivevision.acquisition import NullCameraDriver, build_frame, new_frame_id
from adaptivevision.common.enums import CameraKind
from adaptivevision.common.errors import AcquisitionError
from adaptivevision.common.interfaces import CameraDriver
from adaptivevision.config import CameraConfig


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
