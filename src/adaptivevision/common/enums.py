"""Closed enumerations shared across AdaptiveVision.

All enumerations use explicit, stable string values via :class:`enum.StrEnum`.
The string values are part of the system contract: they are persisted to the
database, encoded into PLC result codes, and rendered on the dashboard. They
must not be reordered or renamed without a coordinated change across every
consumer.
"""

from __future__ import annotations

from enum import StrEnum


class Verdict(StrEnum):
    """Overall outcome of an inspection."""

    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"


class StationState(StrEnum):
    """Lifecycle states of the inspection station (Architecture Spec v1.0 §18)."""

    INIT = "init"
    SELF_TEST = "self_test"
    IDLE = "idle"
    CALIBRATION = "calibration"
    RECIPE_LOADING = "recipe_loading"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    FAULT = "fault"
    ESTOP = "estop"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class Severity(StrEnum):
    """Severity ranking for a detected defect."""

    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class DefectClass(StrEnum):
    """Taxonomy of defect categories.

    The set is extended by later milestones (supervised classification and
    classical AOI). Existing values are stable and must not change, since they
    are shared by the decision policy, PLC codes, and the dashboard.
    """

    DIMENSIONAL = "dimensional"
    ANOMALY = "anomaly"
    SCRATCH = "scratch"
    CONTAMINATION = "contamination"
    MISALIGNMENT = "misalignment"
    MISSING_COMPONENT = "missing_component"
    UNKNOWN = "unknown"


class CameraKind(StrEnum):
    """Supported camera / sensor modalities."""

    AREA_SCAN_2D = "area_scan_2d"
    LINE_SCAN_2D = "line_scan_2d"
    STRUCTURED_LIGHT_3D = "structured_light_3d"
    LASER_TRIANGULATION_3D = "laser_triangulation_3d"
    STEREO_3D = "stereo_3d"
    FILE_REPLAY = "file_replay"


class ExecutionProvider(StrEnum):
    """ONNX Runtime execution providers, listed in typical preference order."""

    TENSORRT = "tensorrt"
    CUDA = "cuda"
    OPENVINO = "openvino"
    CPU = "cpu"
