"""Validated station configuration (Milestone M2).

This module owns the *shape* of the station's runtime configuration and the
validation rules applied to it. Per the frozen architecture, configuration is a
cross-cutting concern: the composition root (Milestone M3) reads raw values from
the environment and builds a validated :class:`StationConfig` here.

The configuration is deliberately small at M2. Later milestones extend it with
camera, PLC, MQTT, and model settings as those subsystems are implemented; the
loading mechanism and validation style established here are reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from adaptivevision.common.enums import CameraKind, ExecutionProvider


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Configuration for a single camera device.

    Attributes:
        camera_id: Stable identifier of the camera.
        kind: Sensor modality.
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        fps: Requested capture rate in frames per second.
        device: Backend-specific device reference (for example a serial number
            or index), if any.
    """

    camera_id: str
    kind: CameraKind
    width: int
    height: int
    fps: float
    device: str | None = None

    def __post_init__(self) -> None:
        """Validate numeric invariants."""
        if self.width <= 0:
            msg = "CameraConfig.width must be positive"
            raise ValueError(msg)
        if self.height <= 0:
            msg = "CameraConfig.height must be positive"
            raise ValueError(msg)
        if self.fps <= 0.0:
            msg = "CameraConfig.fps must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StationConfig:
    """Validated, immutable station configuration.

    Attributes:
        station_id: Stable identifier of this station.
        log_level: Root logging level name (for example ``"INFO"``).
        cameras: Configured camera devices, keyed by ``camera_id``.
        default_recipe_id: Recipe to load at startup, if any.
        execution_provider: Preferred ONNX Runtime execution provider.
        cycle_timeout_ms: Maximum allowed inspection cycle time in milliseconds.
        extra: Unrecognized settings preserved for forward compatibility.
    """

    station_id: str
    log_level: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    default_recipe_id: str | None = None
    execution_provider: ExecutionProvider = ExecutionProvider.CPU
    cycle_timeout_ms: float = 5000.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants."""
        if not self.station_id:
            msg = "StationConfig.station_id must not be empty"
            raise ValueError(msg)
        if self.cycle_timeout_ms <= 0.0:
            msg = "StationConfig.cycle_timeout_ms must be positive"
            raise ValueError(msg)

    def camera(self, camera_id: str) -> CameraConfig:
        """Return the camera with ``camera_id``.

        Args:
            camera_id: Identifier of the camera to look up.

        Returns:
            The matching :class:`CameraConfig`.

        Raises:
            KeyError: If no camera with ``camera_id`` is configured.
        """
        return self.cameras[camera_id]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "station_id": self.station_id,
            "log_level": self.log_level,
            "cameras": {cid: _camera_to_dict(c) for cid, c in self.cameras.items()},
            "default_recipe_id": self.default_recipe_id,
            "execution_provider": self.execution_provider.value,
            "cycle_timeout_ms": self.cycle_timeout_ms,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        cameras = {
            cid: CameraConfig(
                camera_id=c["camera_id"],
                kind=CameraKind(c["kind"]),
                width=c["width"],
                height=c["height"],
                fps=c["fps"],
                device=c.get("device"),
            )
            for cid, c in data.get("cameras", {}).items()
        }
        return cls(
            station_id=data["station_id"],
            log_level=data["log_level"],
            cameras=cameras,
            default_recipe_id=data.get("default_recipe_id"),
            execution_provider=ExecutionProvider(data.get("execution_provider", "cpu")),
            cycle_timeout_ms=data.get("cycle_timeout_ms", 5000.0),
            extra=dict(data.get("extra", {})),
        )


def _camera_to_dict(camera: CameraConfig) -> dict[str, Any]:
    """Serialize a single :class:`CameraConfig`."""
    return {
        "camera_id": camera.camera_id,
        "kind": camera.kind.value,
        "width": camera.width,
        "height": camera.height,
        "fps": camera.fps,
        "device": camera.device,
    }
