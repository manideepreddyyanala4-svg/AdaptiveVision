"""Unit tests for M5 calibration loading and rectification."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from adaptivevision.camera import (
    CalibrationRectifier,
    CameraCalibration,
    identity_calibration,
    load_calibration,
)
from adaptivevision.common import CalibrationError
from adaptivevision.common import RawFrame


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


def _frame(camera_id: str = "cam0", shape: tuple[int, int] = (3, 4)) -> RawFrame:
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
    frame = _frame()
    rectified = CalibrationRectifier(calibration).apply(frame)
    assert rectified.calibration_ver == "calib-v1"
    assert rectified.frame_id == frame.frame_id
    assert rectified.image is not frame.image
    np.testing.assert_array_equal(rectified.image, frame.image)


def test_rectifier_rejects_camera_mismatch() -> None:
    calibration = CameraCalibration.from_dict(_artifact())
    with pytest.raises(CalibrationError, match="does not match"):
        CalibrationRectifier(calibration).apply(_frame(camera_id="other"))


def test_rectifier_rejects_dimension_mismatch() -> None:
    calibration = CameraCalibration.from_dict(_artifact())
    with pytest.raises(CalibrationError, match="do not match"):
        CalibrationRectifier(calibration).apply(_frame(shape=(2, 4)))
