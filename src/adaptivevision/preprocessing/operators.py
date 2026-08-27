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
        normalized = ((image - min_value) * (255.0 / (max_value - min_value))).astype(
            np.uint8
        )
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
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(
        image.dtype
    )
    return _replace_image(frame, gray)


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
