"""Camera calibration and pixel-to-mm foundation (Milestones M5/M16)."""

from adaptivevision.calibration.lifecycle import (
    CalibrationManager,
    CalibrationSelfTest,
    SelfTestResult,
)
from adaptivevision.calibration.model import CameraCalibration, identity_calibration
from adaptivevision.calibration.rectification import CalibrationRectifier
from adaptivevision.calibration.store import load_calibration

__all__ = [
    "CalibrationManager",
    "CalibrationRectifier",
    "CalibrationSelfTest",
    "CameraCalibration",
    "SelfTestResult",
    "identity_calibration",
    "load_calibration",
]
