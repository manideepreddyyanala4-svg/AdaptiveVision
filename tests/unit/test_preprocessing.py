"""Unit tests for M5 preprocessing operators."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from adaptivevision.common.types import RawFrame
from adaptivevision.preprocessing import (
    PreprocessingPipeline,
    ensure_grayscale,
    normalize_uint8,
)


def _frame(image: np.ndarray) -> RawFrame:
    return RawFrame(
        image=image,
        camera_id="cam0",
        frame_id="frame-1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        trigger_id="trigger-1",
    )


def test_normalize_uint8_scales_image() -> None:
    frame = _frame(np.array([[10, 20], [30, 40]], dtype=np.uint16))
    normalized = normalize_uint8(frame)
    assert normalized.image.dtype == np.uint8
    assert normalized.image.min() == 0
    assert normalized.image.max() == 255
    assert normalized.frame_id == frame.frame_id


def test_normalize_uint8_constant_image_returns_zero() -> None:
    frame = _frame(np.full((2, 2), 5, dtype=np.uint8))
    normalized = normalize_uint8(frame)
    assert np.count_nonzero(normalized.image) == 0


def test_ensure_grayscale_converts_rgb() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    gray = ensure_grayscale(_frame(image))
    assert gray.image.shape == (1, 1)
    assert gray.image[0, 0] == 76


def test_ensure_grayscale_copies_grayscale_input() -> None:
    frame = _frame(np.ones((2, 2), dtype=np.uint8))
    gray = ensure_grayscale(frame)
    assert gray.image is not frame.image
    np.testing.assert_array_equal(gray.image, frame.image)


def test_ensure_grayscale_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="Expected"):
        ensure_grayscale(_frame(np.zeros((1, 1, 2), dtype=np.uint8)))


def test_preprocessing_pipeline_applies_steps_in_order() -> None:
    image = np.array([[[0, 0, 0], [10, 10, 10]]], dtype=np.uint8)
    pipeline = PreprocessingPipeline((ensure_grayscale, normalize_uint8))
    processed = pipeline.apply(_frame(image))
    assert processed.image.tolist() == [[0, 255]]
