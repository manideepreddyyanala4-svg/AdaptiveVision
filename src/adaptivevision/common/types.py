"""Immutable value objects that flow through the inspection pipeline.

Per frozen decisions 4 and 5, every value object is a frozen dataclass with
explicit ``to_dict`` / ``from_dict`` serialization (JSON-friendly primitives).
Per decision 9, this module performs no runtime NumPy import; the frame image
type is resolved only during static type checking.

Serialization is provided for the objects that cross a persistence or messaging
boundary (:class:`ROI`, :class:`Pose`, :class:`Tolerance`,
:class:`MeasurementSpec`, :class:`Measurement`). Frames carry raw pixel buffers
and are transient pipeline objects, so they are intentionally not serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self

from adaptivevision.common.geometry import (
    PoseTuple,
    compose_pose,
    invert_pose,
    transform_point,
)

if TYPE_CHECKING:
    import numpy as np

    #: Alias for a raw image buffer. Runtime NumPy is introduced at Milestone M3.
    Image = np.ndarray[Any, np.dtype[Any]]


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

    This is the specification a recipe stores (Milestone M2) and a metrology
    tool evaluates against (Milestone M7). It provides only a pure band
    membership predicate; the pass / fail / review *policy* belongs to M7.

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
    tool (Milestone M7); this value object performs no evaluation itself.

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
