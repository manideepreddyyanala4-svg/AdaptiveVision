"""Calibration application and rectification (Milestone M5)."""

from __future__ import annotations

from adaptivevision.calibration.model import CameraCalibration
from adaptivevision.common.errors import CalibrationError
from adaptivevision.common.types import RawFrame, RectifiedFrame


class CalibrationRectifier:
    """Apply a loaded calibration artifact to raw frames.

    M5 preserves deterministic lineage and validates that frames match the
    loaded calibration. Optical undistortion maps can replace the current
    identity image transform later without changing the pipeline contract.
    """

    def __init__(self, calibration: CameraCalibration) -> None:
        """Initialize with a validated calibration artifact."""
        self._calibration = calibration

    @property
    def calibration(self) -> CameraCalibration:
        """Return the active calibration artifact."""
        return self._calibration

    def apply(self, frame: RawFrame) -> RectifiedFrame:
        """Rectify ``frame`` and attach calibration lineage.

        Raises:
            CalibrationError: If the frame belongs to another camera or has
                dimensions different from the calibration artifact.
        """
        if frame.camera_id != self._calibration.camera_id:
            msg = (
                f"Calibration camera {self._calibration.camera_id!r} does not match "
                f"frame camera {frame.camera_id!r}"
            )
            raise CalibrationError(msg)
        height, width = frame.image.shape[:2]
        if (
            width != self._calibration.image_width
            or height != self._calibration.image_height
        ):
            msg = (
                f"Frame dimensions {width}x{height} do not match calibration "
                f"{self._calibration.image_width}x{self._calibration.image_height}"
            )
            raise CalibrationError(msg)
        return RectifiedFrame(
            image=frame.image.copy(),
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            calibration_ver=self._calibration.version,
            timestamp_monotonic=frame.timestamp_monotonic,
            timestamp_utc=frame.timestamp_utc,
            trigger_id=frame.trigger_id,
        )
