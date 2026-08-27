"""Shared image-loading helpers for the training scripts and dashboard."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def bgr_to_model_input(bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a BGR array (as OpenCV decodes it) to the model's ``(3, H, W)`` contract.

    Returns:
        A ``(3, height, width)`` float32 array with values in ``[0, 255]``,
        matching the channel-first, no-batch-dim convention consumed by
        ``ThresholdAnomalyDetector``.
    """
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32)


def load_rgb(path: Path, height: int, width: int) -> np.ndarray:
    """Read ``path`` as BGR and convert to the model's ``(3, height, width)`` contract."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {path}"
        raise FileNotFoundError(msg)
    return bgr_to_model_input(image, height, width)
