"""Deterministic image preprocessing operators (Milestone M5)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

from adaptivevision.common.types import RawFrame

PreprocessStep: TypeAlias = Callable[[RawFrame], RawFrame]


def normalize_uint8(frame: RawFrame) -> RawFrame:
    """Scale an image to the full uint8 range without mutating the input frame."""
    image = frame.image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value == min_value:
        normalized = np.zeros_like(frame.image, dtype=np.uint8)
    else:
        normalized = ((image - min_value) * (255.0 / (max_value - min_value))).astype(np.uint8)
    return _replace_image(frame, normalized)


def ensure_grayscale(frame: RawFrame) -> RawFrame:
    """Convert RGB/RGBA images to grayscale; grayscale inputs pass through as copies."""
    image = frame.image
    if image.ndim == 2:
        return _replace_image(frame, image.copy())
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        msg = "Expected a grayscale, RGB, or RGBA image"
        raise ValueError(msg)
    rgb = image[:, :, :3].astype(np.float32)
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(image.dtype)
    return _replace_image(frame, gray)


def resize_to(height: int, width: int) -> PreprocessStep:
    """Build a step that resizes a frame to a model's fixed ``(height, width)``.

    Uses nearest-neighbor sampling -- adequate for matching an inference
    contract, and dependency-free like the rest of this module (no OpenCV in
    the production package; that stays a training-only dependency).

    Args:
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        A preprocessing step producing a ``(height, width[, channels])`` image.
    """

    def _resize(frame: RawFrame) -> RawFrame:
        image = frame.image
        src_height, src_width = image.shape[0], image.shape[1]
        row_idx = (np.arange(height) * src_height / height).astype(np.intp)
        col_idx = (np.arange(width) * src_width / width).astype(np.intp)
        resized = image[row_idx][:, col_idx]
        return _replace_image(frame, resized)

    return _resize


class PreprocessingPipeline:
    """Apply preprocessing steps in order to a raw frame."""

    def __init__(self, steps: tuple[PreprocessStep, ...] = ()) -> None:
        """Initialize the pipeline."""
        self._steps = steps

    def apply(self, frame: RawFrame) -> RawFrame:
        """Apply all configured preprocessing steps."""
        current = frame
        for step in self._steps:
            current = step(current)
        return current


def _replace_image(frame: RawFrame, image: np.ndarray[Any, np.dtype[Any]]) -> RawFrame:
    """Return ``frame`` metadata with a replacement image."""
    return RawFrame(
        image=image,
        camera_id=frame.camera_id,
        frame_id=frame.frame_id,
        timestamp_monotonic=frame.timestamp_monotonic,
        timestamp_utc=frame.timestamp_utc,
        trigger_id=frame.trigger_id,
    )
