"""Shared foundation: enums, errors, geometry, IDs, timing, value types, result shapes, and seams.

Every other module depends on these. Everything here is either a pure value
(no behavior beyond serialization) or an abstract seam (a boundary concrete
adapters are injected behind at the composition root, see ``app.py``). Per
frozen decisions 4 and 5, every value
object is a frozen dataclass with explicit ``to_dict``/``from_dict``
serialization; :class:`InspectionResult` round-trips losslessly. Per frozen
decision 1, every seam below is an ABC (not a ``Protocol``), giving explicit
subclassing and the "cannot be instantiated directly" guarantee the null-object
strategy relies on. Per frozen decision 3, the geometry helpers have no
third-party dependencies. Per decision 9, no runtime NumPy import happens here;
the image/embedding array types are resolved only during static type checking.
"""

from __future__ import annotations

import abc
import math
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import numpy as np

    #: Alias for a raw image buffer. Runtime NumPy is introduced at Milestone M3.
    Image = np.ndarray[Any, np.dtype[Any]]

    #: Alias for a fixed-length embedding vector (Milestone M19).
    Embedding = np.ndarray[Any, np.dtype[np.float32]]

__all__ = [
    "ROI",
    "AcquisitionError",
    "AdaptiveVisionError",
    "AdvisoryEngine",
    "AdvisoryError",
    "AdvisoryReport",
    "AdvisoryRepository",
    "AnomalyDetector",
    "AnomalyResult",
    "CalibrationError",
    "CameraDriver",
    "CameraKind",
    "ClassicalResult",
    "Clock",
    "CommsError",
    "Deadline",
    "Defect",
    "DefectClass",
    "DefectMeasurement",
    "ExecutionProvider",
    "FaultError",
    "Homography",
    "InferenceEngine",
    "InferenceError",
    "InspectionEvidence",
    "InspectionResult",
    "Inspector",
    "Measurement",
    "MeasurementSpec",
    "MessagePublisher",
    "MetrologyResult",
    "PLCTransport",
    "PartT",
    "PartialResult",
    "Point",
    "Pose",
    "PoseTuple",
    "RawFrame",
    "RecipeError",
    "RecipeStore",
    "RecipeT",
    "RectifiedFrame",
    "ResultRepository",
    "RetrievalError",
    "RetrievalIndex",
    "RetrievalMatch",
    "Severity",
    "StationState",
    "Stopwatch",
    "Tolerance",
    "Verdict",
    "angle_between_deg",
    "apply_homography",
    "compose_pose",
    "deg_to_rad",
    "distance",
    "invert_pose",
    "measure",
    "new_frame_id",
    "new_inspection_id",
    "new_part_id",
    "new_trace_id",
    "normalize_angle_deg",
    "rad_to_deg",
    "rotate_point",
    "transform_point",
    "translate_point",
]

# =============================================================================
# Enumerations
#
# All enumerations use explicit, stable string values via StrEnum. The string
# values are part of the system contract: they are persisted to the database,
# encoded into PLC result codes, and rendered on the dashboard. They must not
# be reordered or renamed without a coordinated change across every consumer.
# =============================================================================


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

    The set is extended over time (supervised classification and classical
    AOI). Existing values are stable and must not change, since they are
    shared by the decision policy, PLC codes, and the dashboard.
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


# =============================================================================
# Errors
#
# Every error derives from AdaptiveVisionError and carries a `recoverable`
# flag -- the contract consumed by the orchestration layer's failure-handling
# matrix and state machine: a recoverable error degrades a single part to
# REVIEW or triggers a retry, while a non-recoverable error drives the station
# to a fault / safe state. This section only *defines and raises* errors; it
# never catches, handles, or logs them.
# =============================================================================


class AdaptiveVisionError(Exception):
    """Base class for all AdaptiveVision errors.

    Attributes:
        message: Human-readable description of the error.
        recoverable: Whether the station can recover without intervention.
    """

    #: Default recoverability for the class, overridable per instance.
    default_recoverable: bool = False

    def __init__(self, message: str, *, recoverable: bool | None = None) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the error.
            recoverable: Explicit override of :attr:`default_recoverable`.
        """
        super().__init__(message)
        self.message = message
        self.recoverable = self.default_recoverable if recoverable is None else recoverable

    @property
    def is_fatal(self) -> bool:
        """Return ``True`` when the error is non-recoverable."""
        return not self.recoverable


class AcquisitionError(AdaptiveVisionError):
    """Image acquisition failure (camera timeout, disconnect, no frame)."""

    default_recoverable = True


class CalibrationError(AdaptiveVisionError):
    """Missing, invalid, or drifted calibration."""

    default_recoverable = False


class InferenceError(AdaptiveVisionError):
    """Inference engine load, warmup, or execution failure."""

    default_recoverable = True


class CommsError(AdaptiveVisionError):
    """Industrial communication failure (PLC / MQTT)."""

    default_recoverable = True


class RecipeError(AdaptiveVisionError):
    """Invalid, missing, or unloadable recipe."""

    default_recoverable = False


class FaultError(AdaptiveVisionError):
    """General station fault requiring intervention."""

    default_recoverable = False


class RetrievalError(AdaptiveVisionError):
    """Historical-defect vector retrieval failure (Milestone M19)."""

    default_recoverable = True


class AdvisoryError(AdaptiveVisionError):
    """Local LLM advisory (root-cause explanation) failure (Milestone M19)."""

    default_recoverable = True


# =============================================================================
# Geometry
#
# Pure-Python 2D geometry primitives for the coordinate math used by
# calibration and metrology. A 2D rigid pose is (x, y, theta_deg) and applies
# as point -> R(theta_deg) @ point + (x, y) (rotate about the origin, then
# translate).
# =============================================================================

#: A 2D point as ``(x, y)``.
Point = tuple[float, float]

#: A 2D rigid pose as ``(x, y, theta_deg)``.
PoseTuple = tuple[float, float, float]

#: A 3x3 homography as a tuple of three row tuples.
Homography = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


def normalize_angle_deg(degrees: float) -> float:
    """Normalize an angle to the half-open range ``[-180, 180)`` degrees."""
    return (degrees + 180.0) % 360.0 - 180.0


def distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_between_deg(a: Point, b: Point) -> float:
    """Return the angle in degrees of the vector from ``a`` to ``b``."""
    return rad_to_deg(math.atan2(b[1] - a[1], b[0] - a[0]))


def rotate_point(point: Point, degrees: float, origin: Point = (0.0, 0.0)) -> Point:
    """Rotate ``point`` about ``origin`` by ``degrees`` (counter-clockwise)."""
    rad = deg_to_rad(degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    x = cos_a * dx - sin_a * dy + origin[0]
    y = sin_a * dx + cos_a * dy + origin[1]
    return (x, y)


def translate_point(point: Point, dx: float, dy: float) -> Point:
    """Translate ``point`` by ``(dx, dy)``."""
    return (point[0] + dx, point[1] + dy)


def transform_point(pose: PoseTuple, point: Point) -> Point:
    """Apply a rigid ``pose`` to ``point`` (rotate about origin, then translate)."""
    px, py, theta = pose
    rotated = rotate_point(point, theta)
    return (rotated[0] + px, rotated[1] + py)


def compose_pose(first: PoseTuple, second: PoseTuple) -> PoseTuple:
    """Compose two poses: the result applies ``first`` and then ``second``.

    For any point ``p``, ``transform_point(compose_pose(first, second), p)``
    equals ``transform_point(second, transform_point(first, p))``.

    Args:
        first: The pose applied first.
        second: The pose applied second.

    Returns:
        The composed pose.
    """
    rotated = rotate_point((first[0], first[1]), second[2])
    x = rotated[0] + second[0]
    y = rotated[1] + second[1]
    theta = normalize_angle_deg(first[2] + second[2])
    return (x, y, theta)


def invert_pose(pose: PoseTuple) -> PoseTuple:
    """Return the inverse of a rigid ``pose``.

    ``compose_pose(pose, invert_pose(pose))`` is the identity pose
    ``(0, 0, 0)`` up to floating-point error.
    """
    x, y, theta = pose
    rx, ry = rotate_point((x, y), -theta)
    return (-rx, -ry, normalize_angle_deg(-theta))


def apply_homography(homography: Homography, point: Point) -> Point:
    """Apply a 3x3 ``homography`` to ``point`` and return the projected point.

    Args:
        homography: A 3x3 projective transform.
        point: The point to project.

    Returns:
        The projected point.

    Raises:
        ValueError: If the projective denominator is zero (degenerate mapping).
    """
    x, y = point
    row0, row1, row2 = homography
    w = row2[0] * x + row2[1] * y + row2[2]
    if w == 0.0:
        msg = "Degenerate homography: projective denominator is zero"
        raise ValueError(msg)
    px = (row0[0] * x + row0[1] * y + row0[2]) / w
    py = (row1[0] * x + row1[1] * y + row1[2]) / w
    return (px, py)


# =============================================================================
# Domain identifiers
#
# Collision-resistant, time-ordered identifiers for inspections, parts,
# frames, and traces. Each identifier is "{prefix}-{ms:013d}-{rand}" where ms
# is the zero-padded Unix time in milliseconds, so lexical sort approximates
# creation order (to millisecond precision) and the random suffix guarantees
# uniqueness within a millisecond.
# =============================================================================

_MS_WIDTH = 13
_RAND_HEX = 8


def _generate(prefix: str, *, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Build a time-ordered identifier.

    Args:
        prefix: Short type prefix (for example ``"insp"``).
        now_ns: Injected wall-clock time in nanoseconds. Defaults to
            :func:`time.time_ns`. Exposed for deterministic testing.
        rand_hex: Injected random hex suffix. Defaults to a fresh UUID4
            fragment. Exposed for deterministic testing.

    Returns:
        The formatted identifier string.
    """
    nanos = time.time_ns() if now_ns is None else now_ns
    millis = nanos // 1_000_000
    suffix = uuid.uuid4().hex[:_RAND_HEX] if rand_hex is None else rand_hex
    return f"{prefix}-{millis:0{_MS_WIDTH}d}-{suffix}"


def new_inspection_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new inspection identifier (prefix ``insp``)."""
    return _generate("insp", now_ns=now_ns, rand_hex=rand_hex)


def new_part_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new part identifier (prefix ``part``)."""
    return _generate("part", now_ns=now_ns, rand_hex=rand_hex)


def new_frame_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new frame identifier (prefix ``frame``)."""
    return _generate("frame", now_ns=now_ns, rand_hex=rand_hex)


def new_trace_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new trace / correlation identifier (prefix ``trace``)."""
    return _generate("trace", now_ns=now_ns, rand_hex=rand_hex)


# =============================================================================
# Timing
#
# Durations and deadlines use time.monotonic, which is immune to wall-clock
# adjustments (NTP, DST). These only *measure* time; enforcement of a latency
# budget is the orchestration layer's job. Every helper accepts an injectable
# clock callable so tests are deterministic and never sleep.
# =============================================================================

Clock = Callable[[], float]


class Stopwatch:
    """Measures elapsed monotonic time from a start point."""

    def __init__(self, clock: Clock = time.monotonic) -> None:
        """Start the stopwatch.

        Args:
            clock: Monotonic time source in seconds. Defaults to
                :func:`time.monotonic`.
        """
        self._clock = clock
        self._start = clock()

    def reset(self) -> None:
        """Restart the stopwatch from the current time."""
        self._start = self._clock()

    def elapsed_s(self) -> float:
        """Return elapsed time in seconds."""
        return self._clock() - self._start

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed_s() * 1000.0


class Deadline:
    """Tracks a fixed time budget measured against a monotonic clock."""

    def __init__(self, budget_s: float, clock: Clock = time.monotonic) -> None:
        """Create a deadline ``budget_s`` seconds from now.

        Args:
            budget_s: Time budget in seconds.
            clock: Monotonic time source in seconds. Defaults to
                :func:`time.monotonic`.
        """
        self._clock = clock
        self._deadline = clock() + budget_s

    @classmethod
    def from_ms(cls, budget_ms: float, clock: Clock = time.monotonic) -> Deadline:
        """Create a deadline from a budget expressed in milliseconds."""
        return cls(budget_ms / 1000.0, clock=clock)

    def remaining_s(self) -> float:
        """Return remaining time in seconds (negative once expired)."""
        return self._deadline - self._clock()

    def expired(self) -> bool:
        """Return ``True`` once the budget has elapsed."""
        return self._clock() >= self._deadline


@contextmanager
def measure(clock: Clock = time.monotonic) -> Iterator[Stopwatch]:
    """Context manager yielding a :class:`Stopwatch` for the enclosed block.

    Args:
        clock: Monotonic time source in seconds. Defaults to
            :func:`time.monotonic`.

    Yields:
        A running :class:`Stopwatch`; read ``elapsed_ms()`` after the block.
    """
    stopwatch = Stopwatch(clock)
    yield stopwatch


# =============================================================================
# Value types
#
# Immutable value objects that flow through the inspection pipeline. Every
# value object is a frozen dataclass with explicit to_dict/from_dict
# serialization for the ones that cross a persistence or messaging boundary
# (ROI, Pose, Tolerance, MeasurementSpec, Measurement). Frames carry raw pixel
# buffers and are transient pipeline objects, so they are intentionally not
# serializable.
# =============================================================================


@dataclass(frozen=True, slots=True)
class ROI:
    """A rectangular region of interest in a named coordinate frame.

    Attributes:
        label: Human-readable identifier for the region.
        x: X coordinate of the top-left corner.
        y: Y coordinate of the top-left corner.
        width: Region width (non-negative).
        height: Region height (non-negative).
        angle_deg: Rotation of the region in degrees (default ``0``).
    """

    label: str
    x: float
    y: float
    width: float
    height: float
    angle_deg: float = 0.0

    def __post_init__(self) -> None:
        """Validate geometric invariants."""
        if self.width < 0.0 or self.height < 0.0:
            msg = "ROI width and height must be non-negative"
            raise ValueError(msg)

    @property
    def center(self) -> tuple[float, float]:
        """Return the region center as ``(x, y)`` (ignoring rotation)."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "angle_deg": self.angle_deg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            label=data["label"],
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            angle_deg=data.get("angle_deg", 0.0),
        )


@dataclass(frozen=True, slots=True)
class Pose:
    """A 2D rigid pose ``(x, y, theta_deg)``."""

    x: float
    y: float
    theta_deg: float

    def as_tuple(self) -> PoseTuple:
        """Return the pose as a plain ``(x, y, theta_deg)`` tuple."""
        return (self.x, self.y, self.theta_deg)

    def compose(self, other: Pose) -> Pose:
        """Return the pose that applies ``self`` first and then ``other``."""
        x, y, theta = compose_pose(self.as_tuple(), other.as_tuple())
        return Pose(x, y, theta)

    def inverse(self) -> Pose:
        """Return the inverse pose."""
        x, y, theta = invert_pose(self.as_tuple())
        return Pose(x, y, theta)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Apply this pose to ``point``."""
        return transform_point(self.as_tuple(), point)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {"x": self.x, "y": self.y, "theta_deg": self.theta_deg}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(x=data["x"], y=data["y"], theta_deg=data["theta_deg"])


@dataclass(frozen=True, slots=True)
class Tolerance:
    """A tolerance band expressed as deviations from a nominal value.

    A ``None`` bound means unbounded on that side (a unilateral tolerance).

    Attributes:
        minus: Allowed downward deviation (non-negative), or ``None``.
        plus: Allowed upward deviation (non-negative), or ``None``.
    """

    minus: float | None
    plus: float | None

    def __post_init__(self) -> None:
        """Validate that provided deviations are non-negative."""
        if self.minus is not None and self.minus < 0.0:
            msg = "Tolerance.minus must be non-negative"
            raise ValueError(msg)
        if self.plus is not None and self.plus < 0.0:
            msg = "Tolerance.plus must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {"minus": self.minus, "plus": self.plus}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(minus=data["minus"], plus=data["plus"])


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """A named nominal value with a tolerance band and unit.

    This is the specification a recipe stores and a metrology tool evaluates
    against. It provides only a pure band membership predicate; the pass /
    fail / review *policy* belongs to the decision module.

    Attributes:
        name: Identifier of the measured feature.
        nominal: Target value.
        tolerance: Allowed deviation band around ``nominal``.
        unit: Unit of ``nominal`` and the measured value (for example ``"mm"``).
    """

    name: str
    nominal: float
    tolerance: Tolerance
    unit: str

    @property
    def lower(self) -> float | None:
        """Return the absolute lower bound, or ``None`` if unbounded below."""
        if self.tolerance.minus is None:
            return None
        return self.nominal - self.tolerance.minus

    @property
    def upper(self) -> float | None:
        """Return the absolute upper bound, or ``None`` if unbounded above."""
        if self.tolerance.plus is None:
            return None
        return self.nominal + self.tolerance.plus

    def contains(self, value: float) -> bool:
        """Return ``True`` if ``value`` lies within the tolerance band."""
        lower = self.lower
        upper = self.upper
        if lower is not None and value < lower:
            return False
        return not (upper is not None and value > upper)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "nominal": self.nominal,
            "tolerance": self.tolerance.to_dict(),
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            name=data["name"],
            nominal=data["nominal"],
            tolerance=Tolerance.from_dict(data["tolerance"]),
            unit=data["unit"],
        )


@dataclass(frozen=True, slots=True)
class Measurement:
    """A realized measurement value.

    ``in_tolerance`` is a recorded outcome supplied by the producing metrology
    tool; this value object performs no evaluation itself.

    Attributes:
        name: Identifier of the measured feature.
        value: Measured value in ``unit``.
        unit: Unit of ``value``.
        spec: Specification the value was measured against, if any.
        in_tolerance: Recorded pass / fail against ``spec``, if evaluated.
    """

    name: str
    value: float
    unit: str
    spec: MeasurementSpec | None = None
    in_tolerance: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "spec": self.spec.to_dict() if self.spec is not None else None,
            "in_tolerance": self.in_tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        spec_data = data.get("spec")
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data["unit"],
            spec=(MeasurementSpec.from_dict(spec_data) if spec_data is not None else None),
            in_tolerance=data.get("in_tolerance"),
        )


@dataclass(frozen=True, slots=True)
class RawFrame:
    """An acquired, uncorrected image with acquisition metadata.

    The ``image`` buffer is excluded from equality and hashing so frames compare
    by identity metadata rather than by pixel content.

    Attributes:
        image: Raw image buffer.
        camera_id: Identifier of the source camera.
        frame_id: Unique identifier of this frame.
        timestamp_monotonic: Monotonic acquisition time in seconds.
        timestamp_utc: Wall-clock acquisition time (timezone-aware, UTC).
        trigger_id: Identifier of the trigger that produced the frame, if any.
    """

    image: Image = field(compare=False)
    camera_id: str
    frame_id: str
    timestamp_monotonic: float
    timestamp_utc: datetime
    trigger_id: str | None = None


@dataclass(frozen=True, slots=True)
class RectifiedFrame:
    """A frame after optical correction (undistortion and metric rectification).

    Attributes:
        image: Rectified image buffer.
        camera_id: Identifier of the source camera.
        frame_id: Identifier carried over from the source :class:`RawFrame`.
        calibration_ver: Version of the calibration applied.
        timestamp_monotonic: Monotonic acquisition time in seconds.
        timestamp_utc: Wall-clock acquisition time (timezone-aware, UTC).
        trigger_id: Identifier of the originating trigger, if any.
    """

    image: Image = field(compare=False)
    camera_id: str
    frame_id: str
    calibration_ver: str
    timestamp_monotonic: float
    timestamp_utc: datetime
    trigger_id: str | None = None


# =============================================================================
# Result shapes
#
# Result value objects: defects, partial results, and the aggregate result.
# This section defines only the *shapes* of results; the logic that produces
# them lives in metrology.py, decision.py, and the inspection pipeline.
# =============================================================================


@dataclass(frozen=True, slots=True)
class Defect:
    """A single detected defect.

    Attributes:
        defect_class: Category of the defect.
        severity: Severity ranking.
        score: Optional confidence or anomaly score in ``[0, 1]``.
        roi: Optional region the defect was localized to.
        description: Optional human-readable note.
    """

    defect_class: DefectClass
    severity: Severity
    score: float | None = None
    roi: ROI | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "defect_class": self.defect_class.value,
            "severity": self.severity.value,
            "score": self.score,
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        roi_data = data.get("roi")
        return cls(
            defect_class=DefectClass(data["defect_class"]),
            severity=Severity(data["severity"]),
            score=data.get("score"),
            roi=ROI.from_dict(roi_data) if roi_data is not None else None,
            description=data.get("description"),
        )


@dataclass(frozen=True, slots=True)
class DefectMeasurement:
    """A physically-measured defect region extracted from an anomaly heatmap (Milestone M21).

    Produced by :func:`adaptivevision.metrology.measure_defects` from one
    connected region of a thresholded anomaly heatmap -- shape data, not a
    pass/fail judgment (that's still the decision policy's job).

    Attributes:
        bbox: Pixel-space bounding box as ``(x, y, width, height)``, ``x``/``y``
            being the top-left corner.
        area_px2: Region area in pixels.
        area_um2: Region area in square microns, via a caller-supplied
            pixel-to-micron calibration factor.
        aspect_ratio: Bounding-box long-side / short-side ratio, always
            ``>= 1.0``.
        morphology: Coarse shape classification --
            :data:`~adaptivevision.metrology.SCRATCH` when ``aspect_ratio``
            exceeds the elongation threshold, otherwise
            :data:`~adaptivevision.metrology.PARTICLE`.
    """

    bbox: tuple[int, int, int, int]
    area_px2: int
    area_um2: float
    aspect_ratio: float
    morphology: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "bbox": list(self.bbox),
            "area_px2": self.area_px2,
            "area_um2": self.area_um2,
            "aspect_ratio": self.aspect_ratio,
            "morphology": self.morphology,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        bbox = data["bbox"]
        return cls(
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            area_px2=data["area_px2"],
            area_um2=data["area_um2"],
            aspect_ratio=data["aspect_ratio"],
            morphology=data["morphology"],
        )


class PartialResult(abc.ABC):
    """Base type for the output of a single inspector."""

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""


@dataclass(frozen=True, slots=True)
class MetrologyResult(PartialResult):
    """Output of the dimensional metrology inspector.

    Attributes:
        measurements: Measured features with recorded tolerance outcomes.
        defects: Dimensional defects raised by the inspector.
    """

    measurements: tuple[Measurement, ...] = ()
    defects: tuple[Defect, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "measurements": [m.to_dict() for m in self.measurements],
            "defects": [d.to_dict() for d in self.defects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            measurements=tuple(Measurement.from_dict(m) for m in data["measurements"]),
            defects=tuple(Defect.from_dict(d) for d in data["defects"]),
        )


@dataclass(frozen=True, slots=True)
class AnomalyResult(PartialResult):
    """Output of the anomaly inspector.

    Attributes:
        score: Anomaly score for the part.
        threshold: Decision threshold the score was compared against.
        is_anomalous: Whether the detector flagged the part as anomalous.
        heatmap_ref: Optional reference to the archived anomaly heatmap.
        defects: Anomaly defects raised by the inspector.
        defect_measurements: Optional per-region shape measurements extracted
            from the anomaly heatmap (Milestone M21), empty when metrology
            wasn't run.
    """

    score: float
    threshold: float
    is_anomalous: bool
    heatmap_ref: str | None = None
    defects: tuple[Defect, ...] = ()
    defect_measurements: tuple[DefectMeasurement, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "score": self.score,
            "threshold": self.threshold,
            "is_anomalous": self.is_anomalous,
            "heatmap_ref": self.heatmap_ref,
            "defects": [d.to_dict() for d in self.defects],
            "defect_measurements": [m.to_dict() for m in self.defect_measurements],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            score=data["score"],
            threshold=data["threshold"],
            is_anomalous=data["is_anomalous"],
            heatmap_ref=data.get("heatmap_ref"),
            defects=tuple(Defect.from_dict(d) for d in data["defects"]),
            defect_measurements=tuple(
                DefectMeasurement.from_dict(m) for m in data.get("defect_measurements", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ClassicalResult(PartialResult):
    """Output of the classical AOI inspector.

    Attributes:
        defects: Defects raised by classical checks.
    """

    defects: tuple[Defect, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {"defects": [d.to_dict() for d in self.defects]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(defects=tuple(Defect.from_dict(d) for d in data["defects"]))


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """The complete, traceable result of inspecting one part.

    Carries the full lineage (recipe / model / calibration versions) required
    for traceability. Round-trips losslessly via :meth:`to_dict` /
    :meth:`from_dict`.

    Attributes:
        inspection_id: Unique identifier of this inspection.
        part_id: Identifier of the inspected part.
        station_id: Identifier of the station that produced the result.
        verdict: Final verdict.
        recipe_ver: Version of the active recipe.
        model_ver: Version of the anomaly model, if any.
        calib_ver: Version of the calibration applied.
        cycle_time_ms: End-to-end inspection time in milliseconds.
        timestamp_utc: Completion time (timezone-aware, UTC).
        measurements: Measured features.
        defects: Detected defects.
        anomaly_score: Overall anomaly score, if computed.
        image_refs: References to archived images for this part.
        defect_measurements: Heatmap-derived shape measurements (Milestone
            M21), empty when metrology wasn't run.
        drift_status: Sensor/illumination drift status in effect at the time
            of this inspection (:data:`~adaptivevision.drift.SENSOR_DRIFT_ALERT`
            or :data:`~adaptivevision.drift.NOMINAL`), or ``None`` when no
            drift detector was wired in.
    """

    inspection_id: str
    part_id: str
    station_id: str
    verdict: Verdict
    recipe_ver: str
    model_ver: str
    calib_ver: str
    cycle_time_ms: float
    timestamp_utc: datetime
    measurements: tuple[Measurement, ...] = ()
    defects: tuple[Defect, ...] = ()
    anomaly_score: float | None = None
    image_refs: tuple[str, ...] = field(default_factory=tuple)
    defect_measurements: tuple[DefectMeasurement, ...] = ()
    drift_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "inspection_id": self.inspection_id,
            "part_id": self.part_id,
            "station_id": self.station_id,
            "verdict": self.verdict.value,
            "recipe_ver": self.recipe_ver,
            "model_ver": self.model_ver,
            "calib_ver": self.calib_ver,
            "cycle_time_ms": self.cycle_time_ms,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "measurements": [m.to_dict() for m in self.measurements],
            "defects": [d.to_dict() for d in self.defects],
            "anomaly_score": self.anomaly_score,
            "image_refs": list(self.image_refs),
            "defect_measurements": [m.to_dict() for m in self.defect_measurements],
            "drift_status": self.drift_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            inspection_id=data["inspection_id"],
            part_id=data["part_id"],
            station_id=data["station_id"],
            verdict=Verdict(data["verdict"]),
            recipe_ver=data["recipe_ver"],
            model_ver=data["model_ver"],
            calib_ver=data["calib_ver"],
            cycle_time_ms=data["cycle_time_ms"],
            timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]),
            measurements=tuple(Measurement.from_dict(m) for m in data["measurements"]),
            defects=tuple(Defect.from_dict(d) for d in data["defects"]),
            anomaly_score=data.get("anomaly_score"),
            image_refs=tuple(data.get("image_refs", ())),
            defect_measurements=tuple(
                DefectMeasurement.from_dict(m) for m in data.get("defect_measurements", ())
            ),
            drift_status=data.get("drift_status"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    """A single historical-defect match returned by a retrieval index (Milestone M19).

    Attributes:
        vector_id: Identifier of the matched vector within its index.
        distance: Raw metric-space distance or similarity score (metric-specific;
            not normalized across index types).
        dataset: Name of the dataset the match was sourced from.
        category: Product/part category of the match.
        defect_type: Defect type label of the match.
        image_path: Optional path or reference to the matched image.
        metadata: Additional index-specific metadata, JSON-serializable.
    """

    vector_id: int
    distance: float
    dataset: str
    category: str
    defect_type: str
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "vector_id": self.vector_id,
            "distance": self.distance,
            "dataset": self.dataset,
            "category": self.category,
            "defect_type": self.defect_type,
            "image_path": self.image_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            vector_id=data["vector_id"],
            distance=data["distance"],
            dataset=data["dataset"],
            category=data["category"],
            defect_type=data["defect_type"],
            image_path=data.get("image_path"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class InspectionEvidence:
    """Deterministic evidence gathered for one inspection (Milestone M19).

    This is the *only* input the advisory layer (:class:`AdvisoryEngine`) may
    use to produce an explanation. ``severity`` is read from the inspection's
    own :class:`Defect` records (via the decision policy) and is never
    computed or overridden by the advisory layer.

    Attributes:
        sample_id: Identifier of the inspected sample (``inspection_id``).
        category: Product/part category being inspected.
        anomaly_score: Overall anomaly score, if computed.
        severity: Deterministic severity established by the decision policy.
        model_ver: Version of the anomaly model that produced the score.
        retrieval_matches: Historical matches retrieved for this sample.
        heatmap_region: Coarse, computed description of where the anomaly
            concentrates in the image (e.g. ``"upper-right region"``), if a
            per-patch localization signal was available. Real, measured
            evidence like every other field here -- never a place for the
            advisory layer to guess; ``None`` when no localization was run.
    """

    sample_id: str
    category: str
    anomaly_score: float | None
    severity: Severity
    model_ver: str
    retrieval_matches: tuple[RetrievalMatch, ...] = ()
    heatmap_region: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "anomaly_score": self.anomaly_score,
            "severity": self.severity.value,
            "model_ver": self.model_ver,
            "retrieval_matches": [m.to_dict() for m in self.retrieval_matches],
            "heatmap_region": self.heatmap_region,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            sample_id=data["sample_id"],
            category=data["category"],
            anomaly_score=data.get("anomaly_score"),
            severity=Severity(data["severity"]),
            model_ver=data["model_ver"],
            retrieval_matches=tuple(
                RetrievalMatch.from_dict(m) for m in data.get("retrieval_matches", ())
            ),
            heatmap_region=data.get("heatmap_region"),
        )


@dataclass(frozen=True, slots=True)
class AdvisoryReport:
    """A validated advisory (root-cause explanation) report (Milestone M19).

    ``severity`` is always echoed unchanged from the :class:`InspectionEvidence`
    that produced this report; the advisory layer explains the deterministic
    result, it never sets or overrides it.

    Attributes:
        defect_classification: The advisory layer's descriptive classification.
        severity: Deterministic severity, echoed from the evidence.
        confidence_score: Confidence in the hypothesis, in ``[0, 1]``.
        root_cause_hypothesis: Explanatory hypothesis (not a fact).
        recommended_actions: Suggested next steps.
        is_fallback: ``True`` if produced without a live LLM call.
    """

    defect_classification: str
    severity: Severity
    confidence_score: float
    root_cause_hypothesis: str
    recommended_actions: tuple[str, ...] = ()
    is_fallback: bool = False

    def __post_init__(self) -> None:
        """Validate that ``confidence_score`` lies in ``[0, 1]``."""
        if not 0.0 <= self.confidence_score <= 1.0:
            msg = "AdvisoryReport.confidence_score must be in [0, 1]"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "defect_classification": self.defect_classification,
            "severity": self.severity.value,
            "confidence_score": self.confidence_score,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "recommended_actions": list(self.recommended_actions),
            "is_fallback": self.is_fallback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            defect_classification=data["defect_classification"],
            severity=Severity(data["severity"]),
            confidence_score=data["confidence_score"],
            root_cause_hypothesis=data["root_cause_hypothesis"],
            recommended_actions=tuple(data.get("recommended_actions", ())),
            is_fallback=data.get("is_fallback", False),
        )


# =============================================================================
# Abstraction seams
#
# These abstract base classes are the boundaries the domain and orchestration
# layers depend on; concrete adapters are injected at the composition root
# (app.py). Two seams consume aggregates defined elsewhere (the aligned part
# from calibration.py, the recipe from config.py) and are declared generic
# over PartT/RecipeT so the contract is fixed here without a circular import.
# =============================================================================

#: The aligned-part input to an inspector.
PartT = TypeVar("PartT")

#: The recipe aggregate.
RecipeT = TypeVar("RecipeT")


class CameraDriver(abc.ABC):
    """Seam for image acquisition devices.

    Implementations are not required to be thread-safe; a single acquisition
    thread owns the driver.
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Open the device and prepare it for capture.

        Raises:
            AcquisitionError: If the device cannot be opened.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device and its resources."""

    @abc.abstractmethod
    def capture(self, trigger_id: str | None = None) -> RawFrame:
        """Capture a single frame.

        Args:
            trigger_id: Identifier of the triggering event, if any.

        Returns:
            The acquired frame.

        Raises:
            AcquisitionError: On timeout, disconnect, or capture failure.
        """

    @abc.abstractmethod
    def is_healthy(self) -> bool:
        """Return ``True`` if the device is connected and operational."""


class InferenceEngine(abc.ABC):
    """Seam for a model inference backend.

    Manages the lifecycle of a single loaded model.
    """

    @property
    @abc.abstractmethod
    def model_version(self) -> str:
        """Version identifier of the currently loaded model."""

    @abc.abstractmethod
    def load(self, model_id: str) -> None:
        """Load and prepare a model for inference.

        Args:
            model_id: Registry identifier of the model to load.

        Raises:
            InferenceError: If the model cannot be loaded.
        """

    @abc.abstractmethod
    def warmup(self) -> None:
        """Run warmup inferences to stabilize latency.

        Raises:
            InferenceError: If warmup fails.
        """

    @abc.abstractmethod
    def infer(self, inputs: Mapping[str, Image]) -> Mapping[str, Image]:
        """Run inference on named input tensors.

        Args:
            inputs: Mapping of input name to tensor.

        Returns:
            Mapping of output name to tensor.

        Raises:
            InferenceError: On execution failure.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Unload the model and free its resources."""


class AnomalyDetector(abc.ABC):
    """Seam for an anomaly-detection inspector backend."""

    @abc.abstractmethod
    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        """Score a frame (optionally restricted to a region) for anomalies.

        Args:
            frame: The rectified frame to analyze.
            roi: Optional region to restrict analysis to.

        Returns:
            The anomaly result including score and optional heatmap reference.

        Raises:
            InferenceError: On inference failure.
        """


class Inspector(abc.ABC, Generic[PartT, RecipeT]):
    """Seam for an inspection stage.

    Generic over the aligned-part input and the recipe.
    """

    @abc.abstractmethod
    def inspect(self, part: PartT, recipe: RecipeT) -> PartialResult:
        """Inspect an aligned part against a recipe.

        Args:
            part: The localized, aligned part.
            recipe: The active recipe.

        Returns:
            The partial result contributed by this inspector.
        """


class PLCTransport(abc.ABC):
    """Seam for PLC register / coil transport over Modbus TCP."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the transport connection.

        Raises:
            CommsError: If the connection cannot be established.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the transport connection."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return ``True`` if the transport is connected."""

    @abc.abstractmethod
    def read_coils(self, address: int, count: int) -> tuple[bool, ...]:
        """Read ``count`` coils starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """

    @abc.abstractmethod
    def write_coil(self, address: int, value: bool) -> None:
        """Write a single coil.

        Raises:
            CommsError: On communication failure.
        """

    @abc.abstractmethod
    def read_registers(self, address: int, count: int) -> tuple[int, ...]:
        """Read ``count`` holding registers starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """

    @abc.abstractmethod
    def write_registers(self, address: int, values: Sequence[int]) -> None:
        """Write holding registers starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """


class MessagePublisher(abc.ABC):
    """Seam for publishing messages to a broker."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the broker connection.

        Raises:
            CommsError: If the connection cannot be established.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection."""

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return ``True`` if the publisher is connected."""

    @abc.abstractmethod
    def publish(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        """Publish a payload to a topic.

        Args:
            topic: Destination topic.
            payload: JSON-serializable message body.
            qos: Delivery quality-of-service level.
            retain: Whether the broker should retain the message.

        Raises:
            CommsError: On publish failure.
        """


class ResultRepository(abc.ABC):
    """Seam for persisting and querying inspection results."""

    @abc.abstractmethod
    def save_result(self, result: InspectionResult) -> None:
        """Persist an inspection result.

        Raises:
            AdaptiveVisionError: On storage failure.
        """

    @abc.abstractmethod
    def get_result(self, inspection_id: str) -> InspectionResult | None:
        """Return the result with ``inspection_id``, or ``None`` if absent."""

    @abc.abstractmethod
    def list_results(self, *, limit: int = 100, offset: int = 0) -> tuple[InspectionResult, ...]:
        """Return a page of results ordered most-recent first."""


class RecipeStore(abc.ABC, Generic[RecipeT]):
    """Seam for recipe storage and versioning.

    Generic over the recipe aggregate.
    """

    @abc.abstractmethod
    def load(self, recipe_id: str) -> RecipeT:
        """Load a recipe by identifier.

        Raises:
            RecipeError: If the recipe is missing or invalid.
        """

    @abc.abstractmethod
    def save(self, recipe: RecipeT) -> None:
        """Persist a recipe.

        Raises:
            RecipeError: On storage failure.
        """

    @abc.abstractmethod
    def list_ids(self) -> tuple[str, ...]:
        """Return the identifiers of all stored recipes."""


class RetrievalIndex(abc.ABC):
    """Seam for historical-defect vector retrieval (Milestone M19).

    Implementations store embeddings alongside metadata and support
    approximate/exact nearest-neighbor search. The index is not the source of
    truth for business metadata (that remains the persistence layer) - it only
    maps vectors to the metadata needed to identify a historical match.
    """

    @abc.abstractmethod
    def add(self, embeddings: Embedding, metadata: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        """Add embeddings with associated metadata.

        Args:
            embeddings: A 2D array of shape ``(n, dim)``.
            metadata: One mapping per embedding, same length as ``embeddings``.

        Returns:
            The vector IDs assigned to the added embeddings, in order.

        Raises:
            RetrievalError: On dimension mismatch, non-finite values, or a
                length mismatch between ``embeddings`` and ``metadata``.
        """

    @abc.abstractmethod
    def search(self, query: Embedding, top_k: int = 3) -> tuple[RetrievalMatch, ...]:
        """Return the ``top_k`` nearest historical matches to ``query``.

        Args:
            query: A 1D embedding of shape ``(dim,)``.
            top_k: Maximum number of matches to return.

        Raises:
            RetrievalError: On dimension mismatch or search failure.
        """

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist the index and its metadata to ``path``.

        Raises:
            RetrievalError: On storage failure.
        """

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Load a previously saved index and its metadata from ``path``.

        Raises:
            RetrievalError: If the index is missing, corrupt, or incompatible.
        """


class AdvisoryEngine(abc.ABC):
    """Seam for local-LLM advisory root-cause explanation (Milestone M19).

    Implementations MUST treat the deterministic severity carried on
    :class:`InspectionEvidence` as authoritative and only explain it - never
    override, upgrade, or downgrade it. Implementations must never raise for
    an unavailable or misbehaving LLM: they fall back to a deterministic
    report derived only from the supplied evidence.
    """

    @abc.abstractmethod
    def generate_report(self, evidence: InspectionEvidence) -> AdvisoryReport:
        """Produce a validated advisory report explaining ``evidence``."""


class AdvisoryRepository(abc.ABC):
    """Seam for persisting and querying advisory reports (Milestone M19)."""

    @abc.abstractmethod
    def save_report(
        self,
        inspection_id: str,
        evidence: InspectionEvidence,
        report: AdvisoryReport,
    ) -> None:
        """Persist an advisory report linked to ``inspection_id``.

        Raises:
            AdaptiveVisionError: On storage failure.
        """

    @abc.abstractmethod
    def get_report(self, inspection_id: str) -> AdvisoryReport | None:
        """Return the advisory report for ``inspection_id``, or ``None``."""
