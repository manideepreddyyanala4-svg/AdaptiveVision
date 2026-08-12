"""Configuration loading and station settings (Milestone M2).

This package owns the validated, immutable station configuration
(:class:`StationConfig` and :class:`CameraConfig`) and the environment-based
loader (:func:`load_config`) that the composition root (Milestone M3) uses to
bootstrap the station. Configuration is a cross-cutting concern: it is read once
at startup and injected into the subsystems that need it.
"""

from __future__ import annotations

from adaptivevision.config.loader import load_config, load_env_file
from adaptivevision.config.settings import CameraConfig, StationConfig

__all__ = [
    "CameraConfig",
    "StationConfig",
    "load_config",
    "load_env_file",
]
