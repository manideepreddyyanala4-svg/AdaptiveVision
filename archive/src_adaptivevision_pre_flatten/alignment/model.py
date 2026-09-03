"""Alignment domain objects (Milestone M6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from adaptivevision.common.errors import FaultError
from adaptivevision.common.types import Pose, RectifiedFrame


@dataclass(frozen=True, slots=True)
class GoldenReference:
    """Versioned 2D reference used to localize a part.

    Attributes:
        reference_id: Stable identifier of the golden reference.
        version: Version string of this reference artifact.
        camera_id: Camera this reference applies to.
        image_width: Expected rectified image width.
        image_height: Expected rectified image height.
        nominal_pose: Pose of a correctly aligned part in reference space.
        min_score: Minimum alignment score accepted by the aligner.
    """

    reference_id: str
    version: str
    camera_id: str
    image_width: int
    image_height: int
    nominal_pose: Pose = field(default_factory=lambda: Pose(0.0, 0.0, 0.0))
    min_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate reference invariants."""
        if not self.reference_id:
            msg = "GoldenReference.reference_id must not be empty"
            raise FaultError(msg)
        if not self.version:
            msg = "GoldenReference.version must not be empty"
            raise FaultError(msg)
        if not self.camera_id:
            msg = "GoldenReference.camera_id must not be empty"
            raise FaultError(msg)
        if self.image_width <= 0 or self.image_height <= 0:
            msg = "GoldenReference image dimensions must be positive"
            raise FaultError(msg)
        if not 0.0 <= self.min_score <= 1.0:
            msg = "GoldenReference.min_score must be in [0, 1]"
            raise FaultError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "reference_id": self.reference_id,
            "version": self.version,
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "nominal_pose": self.nominal_pose.to_dict(),
            "min_score": self.min_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            reference_id=data["reference_id"],
            version=data["version"],
            camera_id=data["camera_id"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            nominal_pose=Pose.from_dict(
                data.get("nominal_pose", {"x": 0.0, "y": 0.0, "theta_deg": 0.0})
            ),
            min_score=data.get("min_score", 0.0),
        )


@dataclass(frozen=True, slots=True)
class LocalizedPart:
    """A rectified frame localized against a golden reference."""

    frame: RectifiedFrame
    pose: Pose
    reference_id: str
    reference_ver: str
    score: float

    def __post_init__(self) -> None:
        """Validate localized-part invariants."""
        if not 0.0 <= self.score <= 1.0:
            msg = "LocalizedPart.score must be in [0, 1]"
            raise FaultError(msg)

    def lineage(self) -> dict[str, Any]:
        """Return JSON-friendly alignment lineage."""
        return {
            "frame_id": self.frame.frame_id,
            "calibration_ver": self.frame.calibration_ver,
            "reference_id": self.reference_id,
            "reference_ver": self.reference_ver,
            "pose": self.pose.to_dict(),
            "score": self.score,
        }
