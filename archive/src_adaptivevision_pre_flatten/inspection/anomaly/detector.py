"""AI-based anomaly detection inspectors (Milestone M9).

This module provides concrete :class:`AnomalyDetector` implementations that
score a rectified frame and produce an :class:`AnomalyResult`. Following the
M7 metrology pattern, a deterministic static detector is provided for replay,
tests, and simulated stations, and a threshold detector wraps the M8 inference
engine so a real model can be used without coupling the inspector to ONNX.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from adaptivevision.common.enums import DefectClass, Severity
from adaptivevision.common.interfaces import AnomalyDetector, InferenceEngine
from adaptivevision.common.result import AnomalyResult, Defect
from adaptivevision.common.types import ROI, RectifiedFrame

#: Extracts a scalar anomaly score from an inference output mapping.
ScoreExtractor = Callable[[Mapping[str, np.ndarray[Any, np.dtype[Any]]]], float]


class StaticAnomalyDetector(AnomalyDetector):
    """Deterministic anomaly detector for replay, tests, and simulated stations.

    Args:
        score: Fixed anomaly score returned for every frame.
        threshold: Decision threshold the score is compared against.
        heatmap_ref: Optional heatmap reference to attach to the result.
    """

    def __init__(
        self,
        score: float,
        threshold: float,
        *,
        heatmap_ref: str | None = None,
    ) -> None:
        """Initialize the detector."""
        self._score = score
        self._threshold = threshold
        self._heatmap_ref = heatmap_ref

    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        """Return a fixed anomaly result for the frame.

        Args:
            frame: The rectified frame to analyze.
            roi: Optional region to restrict analysis to.

        Returns:
            An :class:`AnomalyResult` with the configured score and threshold.
        """
        _ = (frame, roi)
        is_anomalous = self._score >= self._threshold
        defects: tuple[Defect, ...] = ()
        if is_anomalous:
            defects = (
                Defect(
                    defect_class=DefectClass.ANOMALY,
                    severity=Severity.MAJOR,
                    score=self._score,
                    roi=roi,
                    description="Anomaly score exceeded threshold",
                ),
            )
        return AnomalyResult(
            score=self._score,
            threshold=self._threshold,
            is_anomalous=is_anomalous,
            heatmap_ref=self._heatmap_ref,
            defects=defects,
        )


class ThresholdAnomalyDetector(AnomalyDetector):
    """Score a frame with an inference engine and apply a decision threshold.

    Args:
        engine: The M8 inference engine used to score the frame.
        threshold: Anomaly score threshold; scores at or above it are anomalous.
        input_name: Name of the model input the frame image is fed to.
        output_name: Name of the model output that carries the anomaly score.
        score_extractor: Optional callable that maps the inference output to a
            scalar score. Defaults to reading ``output_name`` and taking the
            first element.
        anomalous_severity: Severity assigned to the defect raised for an
            anomalous frame. A recipe's ``review_on_anomaly`` flag maps to
            ``Severity.MINOR`` here (the M10 decision policy routes MINOR to
            REVIEW); the default ``Severity.MAJOR`` routes straight to FAIL.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        threshold: float,
        *,
        input_name: str = "input",
        output_name: str = "output",
        score_extractor: ScoreExtractor | None = None,
        anomalous_severity: Severity = Severity.MAJOR,
    ) -> None:
        """Initialize the detector."""
        self._engine = engine
        self._threshold = threshold
        self._input_name = input_name
        self._output_name = output_name
        self._score_extractor = score_extractor or self._default_score
        self._anomalous_severity = anomalous_severity

    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        """Score the frame and apply the decision threshold.

        Args:
            frame: The rectified frame to analyze.
            roi: Optional region to restrict analysis to.

        Returns:
            An :class:`AnomalyResult` with the model score and threshold.

        Raises:
            InferenceError: If inference fails.
        """
        _ = roi
        image = frame.image.astype(np.float32, copy=False)
        if image.ndim == 2:
            # Grayscale (H, W) -> (1, H, W): a 1-channel model input.
            image = image[np.newaxis, ...]
        elif image.ndim == 3:
            # Color (H, W, C) -> (C, H, W): the channel-first layout ONNX
            # vision models expect.
            image = image.transpose(2, 0, 1)
        outputs = self._engine.infer({self._input_name: image})
        score = self._score_extractor(outputs)
        is_anomalous = score >= self._threshold
        defects: tuple[Defect, ...] = ()
        if is_anomalous:
            defects = (
                Defect(
                    defect_class=DefectClass.ANOMALY,
                    severity=self._anomalous_severity,
                    score=score,
                    description="Anomaly score exceeded threshold",
                ),
            )
        return AnomalyResult(
            score=score,
            threshold=self._threshold,
            is_anomalous=is_anomalous,
            defects=defects,
        )

    @staticmethod
    def _default_score(
        outputs: Mapping[str, np.ndarray[Any, np.dtype[Any]]],
    ) -> float:
        """Extract a scalar score from the configured output tensor."""
        values = outputs["output"]
        return float(np.asarray(values).reshape(-1)[0])
