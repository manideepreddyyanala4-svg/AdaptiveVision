"""Shared domain contracts for AdaptiveVision (Milestone M1).

This package defines the system's shared vocabulary (value types, enums, error
taxonomy) and the abstraction seams (interfaces) that every later milestone
builds against. It contains contracts, not behaviour. The names below form the
stable public surface of the domain layer.
"""

from __future__ import annotations

from adaptivevision.common.enums import (
    CameraKind,
    DefectClass,
    ExecutionProvider,
    Severity,
    StationState,
    Verdict,
)
from adaptivevision.common.errors import (
    AcquisitionError,
    AdaptiveVisionError,
    CalibrationError,
    CommsError,
    FaultError,
    InferenceError,
    RecipeError,
)
from adaptivevision.common.ids import (
    new_frame_id,
    new_inspection_id,
    new_part_id,
    new_trace_id,
)
from adaptivevision.common.interfaces import (
    AnomalyDetector,
    CameraDriver,
    InferenceEngine,
    Inspector,
    MessagePublisher,
    PLCTransport,
    RecipeStore,
    ResultRepository,
)
from adaptivevision.common.result import (
    AnomalyResult,
    ClassicalResult,
    Defect,
    InspectionResult,
    MetrologyResult,
    PartialResult,
)
from adaptivevision.common.timing import Deadline, Stopwatch, measure
from adaptivevision.common.types import (
    ROI,
    Measurement,
    MeasurementSpec,
    Pose,
    RawFrame,
    RectifiedFrame,
    Tolerance,
)

__all__ = [
    "ROI",
    "AcquisitionError",
    "AdaptiveVisionError",
    "AnomalyDetector",
    "AnomalyResult",
    "CalibrationError",
    "CameraDriver",
    "CameraKind",
    "ClassicalResult",
    "CommsError",
    "Deadline",
    "Defect",
    "DefectClass",
    "ExecutionProvider",
    "FaultError",
    "InferenceEngine",
    "InferenceError",
    "InspectionResult",
    "Inspector",
    "Measurement",
    "MeasurementSpec",
    "MessagePublisher",
    "MetrologyResult",
    "PLCTransport",
    "PartialResult",
    "Pose",
    "RawFrame",
    "RecipeError",
    "RecipeStore",
    "RectifiedFrame",
    "ResultRepository",
    "Severity",
    "StationState",
    "Stopwatch",
    "Tolerance",
    "Verdict",
    "measure",
    "new_frame_id",
    "new_inspection_id",
    "new_part_id",
    "new_trace_id",
]
