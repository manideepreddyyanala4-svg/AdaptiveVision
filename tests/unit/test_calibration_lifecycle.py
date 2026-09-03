"""Unit tests for the M16 calibration lifecycle, hot-swap, and self-test."""

from __future__ import annotations

import pytest

from adaptivevision.camera import (
    CalibrationManager,
    CalibrationSelfTest,
    CameraCalibration,
    identity_calibration,
)
from adaptivevision.common import CalibrationError


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
