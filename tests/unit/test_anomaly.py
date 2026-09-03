"""Unit tests for M9 anomaly detection inspectors."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from adaptivevision.common import DefectClass, Severity
from adaptivevision.common import InferenceEngine
from adaptivevision.common import AnomalyResult
from adaptivevision.common import ROI, RectifiedFrame
from adaptivevision.metrology import (
    StaticAnomalyDetector,
    ThresholdAnomalyDetector,
)


def _frame(image: np.ndarray[Any, np.dtype[Any]]) -> RectifiedFrame:
    return RectifiedFrame(
        image=image,
        camera_id="cam-1",
        frame_id="frame-1",
        calibration_ver="cal-1",
        timestamp_monotonic=1.0,
        timestamp_utc=datetime.now(UTC),
    )


class _FakeEngine(InferenceEngine):
    def __init__(self, score: float) -> None:
        self._score = score
        self._inputs: list[dict[str, np.ndarray[Any, np.dtype[Any]]]] = []

    @property
    def model_version(self) -> str:
        return "model-v1"

    def load(self, model_id: str) -> None:
        _ = model_id

    def warmup(self) -> None:
        return None

    def infer(
        self, inputs: dict[str, np.ndarray[Any, np.dtype[Any]]]
    ) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
        self._inputs.append(inputs)
        return {"output": np.array([self._score], dtype=np.float32)}

    def unload(self) -> None:
        return None


def test_static_detector_below_threshold_is_not_anomalous() -> None:
    detector = StaticAnomalyDetector(score=0.2, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.score == 0.2
    assert result.threshold == 0.5
    assert result.is_anomalous is False
    assert result.defects == ()


def test_static_detector_at_threshold_is_anomalous_with_defect() -> None:
    detector = StaticAnomalyDetector(score=0.5, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.is_anomalous is True
    assert len(result.defects) == 1
    defect = result.defects[0]
    assert defect.defect_class == DefectClass.ANOMALY
    assert defect.severity == Severity.MAJOR
    assert defect.score == 0.5


def test_static_detector_carries_heatmap_ref_and_roi() -> None:
    roi = ROI(label="region", x=0.0, y=0.0, width=2.0, height=2.0)
    detector = StaticAnomalyDetector(
        score=0.9, threshold=0.5, heatmap_ref="heatmaps/part-1.png"
    )
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)), roi=roi)
    assert result.heatmap_ref == "heatmaps/part-1.png"
    assert result.defects[0].roi == roi


def test_threshold_detector_batches_2d_image_and_extracts_score() -> None:
    engine = _FakeEngine(score=0.8)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.score == pytest.approx(0.8)
    assert result.is_anomalous is True
    assert engine._inputs[0]["input"].ndim == 3


def test_threshold_detector_below_threshold_is_clean() -> None:
    engine = _FakeEngine(score=0.1)
    detector = ThresholdAnomalyDetector(engine, threshold=0.5)
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.is_anomalous is False
    assert result.defects == ()


def test_threshold_detector_uses_custom_score_extractor() -> None:
    engine = _FakeEngine(score=0.0)
    detector = ThresholdAnomalyDetector(
        engine,
        threshold=0.5,
        score_extractor=lambda outputs: float(outputs["output"][0]) + 0.9,
    )
    result = detector.detect(_frame(np.zeros((4, 4), dtype=np.uint8)))
    assert result.score == pytest.approx(0.9)
    assert result.is_anomalous is True


def test_anomaly_result_round_trips() -> None:
    result = AnomalyResult(
        score=0.7,
        threshold=0.5,
        is_anomalous=True,
        heatmap_ref="heatmaps/part-1.png",
    )
    restored = AnomalyResult.from_dict(result.to_dict())
    assert restored == result
