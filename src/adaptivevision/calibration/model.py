"""Calibration artifact model (Milestone M5).

Calibration artifacts are immutable, versioned records loaded by the
composition root and applied by the rectification stage. M5 records the
pixel-to-millimeter foundation and optical model metadata; full calibration
lifecycle management and hot-swap behavior belongs to M16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from adaptivevision.common.errors import CalibrationError


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Versioned calibration for a single camera.

    Attributes:
        calibration_id: Stable identifier of the calibration artifact.
        version: Version string recorded into inspection lineage.
        camera_id: Camera this calibration applies to.
        image_width: Calibrated image width in pixels.
        image_height: Calibrated image height in pixels.
        pixel_size_mm: Physical size represented by one pixel.
        intrinsic_matrix: 3x3 camera matrix.
        distortion_coefficients: Lens distortion coefficients.
    """

    calibration_id: str
    version: str
    camera_id: str
    image_width: int
    image_height: int
    pixel_size_mm: float
    intrinsic_matrix: tuple[tuple[float, float, float], ...]
    distortion_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate calibration invariants."""
        if not self.calibration_id:
            msg = "CameraCalibration.calibration_id must not be empty"
            raise CalibrationError(msg)
        if not self.version:
            msg = "CameraCalibration.version must not be empty"
            raise CalibrationError(msg)
        if not self.camera_id:
            msg = "CameraCalibration.camera_id must not be empty"
            raise CalibrationError(msg)
        if self.image_width <= 0 or self.image_height <= 0:
            msg = "CameraCalibration image dimensions must be positive"
            raise CalibrationError(msg)
        if self.pixel_size_mm <= 0.0:
            msg = "CameraCalibration.pixel_size_mm must be positive"
            raise CalibrationError(msg)
        if len(self.intrinsic_matrix) != 3 or any(len(row) != 3 for row in self.intrinsic_matrix):
            msg = "CameraCalibration.intrinsic_matrix must be 3x3"
            raise CalibrationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "calibration_id": self.calibration_id,
            "version": self.version,
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "pixel_size_mm": self.pixel_size_mm,
            "intrinsic_matrix": [list(row) for row in self.intrinsic_matrix],
            "distortion_coefficients": list(self.distortion_coefficients),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize and validate a calibration artifact."""
        return cls(
            calibration_id=data["calibration_id"],
            version=data["version"],
            camera_id=data["camera_id"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            pixel_size_mm=data["pixel_size_mm"],
            intrinsic_matrix=_matrix_from_raw(data["intrinsic_matrix"]),
            distortion_coefficients=tuple(
                float(v) for v in data.get("distortion_coefficients", ())
            ),
        )


def identity_calibration(
    *,
    camera_id: str,
    width: int,
    height: int,
    version: str = "identity",
    pixel_size_mm: float = 1.0,
) -> CameraCalibration:
    """Build an explicit identity calibration for synthetic or uncalibrated runs."""
    return CameraCalibration(
        calibration_id=f"{camera_id}-{version}",
        version=version,
        camera_id=camera_id,
        image_width=width,
        image_height=height,
        pixel_size_mm=pixel_size_mm,
        intrinsic_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _matrix_from_raw(raw: Any) -> tuple[tuple[float, float, float], ...]:
    """Convert a raw JSON matrix into a typed 3x3 tuple."""
    rows = tuple(tuple(float(v) for v in row) for row in raw)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        msg = "CameraCalibration.intrinsic_matrix must be 3x3"
        raise CalibrationError(msg)
    return (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    )
