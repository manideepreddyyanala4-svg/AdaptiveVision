"""Calibration artifact loading (Milestone M5)."""

from __future__ import annotations

import json
from pathlib import Path

from adaptivevision.calibration.model import CameraCalibration
from adaptivevision.common.errors import CalibrationError


def load_calibration(path: str | Path) -> CameraCalibration:
    """Load a calibration artifact from JSON.

    Args:
        path: Filesystem path to the calibration JSON document.

    Returns:
        The validated calibration artifact.

    Raises:
        CalibrationError: If the file cannot be read, parsed, or validated.
    """
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to load calibration artifact {artifact_path}: {exc}"
        raise CalibrationError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Calibration artifact {artifact_path} must contain a JSON object"
        raise CalibrationError(msg)
    try:
        return CameraCalibration.from_dict(payload)
    except (KeyError, TypeError, ValueError, CalibrationError) as exc:
        msg = f"Invalid calibration artifact {artifact_path}: {exc}"
        raise CalibrationError(msg) from exc
