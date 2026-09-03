"""Calibration lifecycle, hot-swap, and self-test (Milestone M16).

The :class:`CalibrationManager` holds the active calibration for each camera
and supports atomic hot-swap of a new calibration artifact. The
:class:`CalibrationSelfTest` validates a calibration artifact before it is
activated.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptivevision.calibration.model import CameraCalibration
from adaptivevision.common.errors import CalibrationError


@dataclass(frozen=True, slots=True)
class SelfTestResult:
    """Outcome of a calibration self-test.

    Attributes:
        passed: Whether the calibration passed all checks.
        checks: Human-readable results of each performed check.
    """

    passed: bool
    checks: tuple[str, ...]


class CalibrationSelfTest:
    """Runs validation checks against a calibration artifact."""

    def run(self, calibration: CameraCalibration) -> SelfTestResult:
        """Validate ``calibration`` and return the self-test outcome.

        Args:
            calibration: The calibration artifact to validate.

        Returns:
            The self-test result.
        """
        checks: list[str] = []
        passed = True

        if calibration.pixel_size_mm > 0.0:
            checks.append("pixel_size_mm positive")
        else:
            checks.append("pixel_size_mm NOT positive")
            passed = False

        if self._matrix_is_invertible(calibration.intrinsic_matrix):
            checks.append("intrinsic_matrix invertible")
        else:
            checks.append("intrinsic_matrix NOT invertible")
            passed = False

        if calibration.image_width > 0 and calibration.image_height > 0:
            checks.append("image dimensions positive")
        else:
            checks.append("image dimensions NOT positive")
            passed = False

        return SelfTestResult(passed=passed, checks=tuple(checks))

    @staticmethod
    def _matrix_is_invertible(
        matrix: tuple[tuple[float, float, float], ...],
    ) -> bool:
        """Return ``True`` if the 3x3 matrix has a non-zero determinant."""
        (a, b, c), (d, e, f), (g, h, i) = matrix
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        return abs(determinant) > 1e-12


class CalibrationManager:
    """Manages the active calibration per camera with hot-swap support.

    The manager keeps the currently active calibration for each camera and
    supports atomic replacement. A new calibration is validated by the
    self-test before it becomes active.
    """

    def __init__(self, self_test: CalibrationSelfTest | None = None) -> None:
        """Initialize an empty manager."""
        self._self_test = self_test or CalibrationSelfTest()
        self._active: dict[str, CameraCalibration] = {}

    def active(self, camera_id: str) -> CameraCalibration | None:
        """Return the active calibration for a camera, or ``None``."""
        return self._active.get(camera_id)

    def activate(self, calibration: CameraCalibration) -> None:
        """Hot-swap ``calibration`` as the active artifact for its camera.

        Args:
            calibration: The calibration to activate.

        Raises:
            CalibrationError: If the calibration fails its self-test.
        """
        result = self._self_test.run(calibration)
        if not result.passed:
            msg = f"Calibration {calibration.calibration_id} failed self-test"
            raise CalibrationError(msg)
        self._active[calibration.camera_id] = calibration

    def cameras(self) -> tuple[str, ...]:
        """Return the camera ids with an active calibration."""
        return tuple(sorted(self._active))
