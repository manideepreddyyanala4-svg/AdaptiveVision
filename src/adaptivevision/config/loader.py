"""Configuration loading from the environment (Milestone M2).

The composition root reads configuration from environment variables and an
optional ``.env`` file. This module provides a small, dependency-free loader
that:

* parses a ``KEY=VALUE`` ``.env`` file (without overriding already-set
  environment variables),
* reads a fixed set of ``ADAPTIVEVISION_*`` variables,
* and builds a validated :class:`~adaptivevision.config.settings.StationConfig`.

Unknown ``ADAPTIVEVISION_*`` variables are preserved in
:attr:`StationConfig.extra` for forward compatibility rather than rejected, so
later milestones can add settings without breaking existing deployments.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from adaptivevision.common.enums import CameraKind, ExecutionProvider
from adaptivevision.config.settings import CameraConfig, StationConfig

#: Prefix for all AdaptiveVision environment variables.
_ENV_PREFIX = "ADAPTIVEVISION_"

#: Default station identifier when none is configured.
_DEFAULT_STATION_ID = "station-01"


def load_env_file(path: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` ``.env`` file into a dictionary.

    Existing environment variables take precedence: a key already present in
    ``environ`` is not overridden by the file. Blank lines and lines starting
    with ``#`` are ignored. Values may be quoted with single or double quotes.

    Args:
        path: Path to the ``.env`` file.
        environ: Environment mapping to consult for precedence. Defaults to
            :data:`os.environ`.

    Returns:
        A dictionary of parsed key/value pairs (only those not already set).
    """
    env = os.environ if environ is None else environ
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in env:
            parsed[key] = value
    return parsed


def load_config(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> StationConfig:
    """Load and validate station configuration.

    Args:
        environ: Environment mapping to read from. Defaults to
            :data:`os.environ`.
        env_file: Optional ``.env`` file to merge in (existing environment
            variables win). Defaults to ``.env`` in the current directory.

    Returns:
        A validated :class:`StationConfig`.

    Raises:
        ValueError: If a configured value is invalid.
    """
    env = os.environ if environ is None else environ
    merged: dict[str, str] = dict(env)
    if env_file is not None:
        merged.update(load_env_file(env_file, environ=env))

    station_id = merged.get(f"{_ENV_PREFIX}STATION_ID", _DEFAULT_STATION_ID)
    log_level = merged.get(f"{_ENV_PREFIX}LOG_LEVEL", "INFO")
    default_recipe_id = merged.get(f"{_ENV_PREFIX}DEFAULT_RECIPE_ID")
    provider_name = merged.get(f"{_ENV_PREFIX}EXECUTION_PROVIDER", "cpu")
    cycle_timeout_ms = _parse_float(
        merged.get(f"{_ENV_PREFIX}CYCLE_TIMEOUT_MS", "5000.0"),
        f"{_ENV_PREFIX}CYCLE_TIMEOUT_MS",
    )

    cameras = _parse_cameras(merged)

    extra = {
        key[len(_ENV_PREFIX) :]: value
        for key, value in merged.items()
        if key.startswith(_ENV_PREFIX)
        and key
        not in {
            f"{_ENV_PREFIX}STATION_ID",
            f"{_ENV_PREFIX}LOG_LEVEL",
            f"{_ENV_PREFIX}DEFAULT_RECIPE_ID",
            f"{_ENV_PREFIX}EXECUTION_PROVIDER",
            f"{_ENV_PREFIX}CYCLE_TIMEOUT_MS",
            f"{_ENV_PREFIX}CAMERA_IDS",
        }
    }

    return StationConfig(
        station_id=station_id,
        log_level=log_level,
        cameras=cameras,
        default_recipe_id=default_recipe_id,
        execution_provider=ExecutionProvider(provider_name),
        cycle_timeout_ms=cycle_timeout_ms,
        extra=extra,
    )


def _parse_cameras(merged: Mapping[str, str]) -> dict[str, CameraConfig]:
    """Parse camera configurations from ``ADAPTIVEVISION_CAMERA_IDS``.

    Each camera id in the comma-separated ``ADAPTIVEVISION_CAMERA_IDS`` list is
    expanded with ``ADAPTIVEVISION_CAMERA_<ID>_*`` variables.

    Args:
        merged: Merged environment mapping.

    Returns:
        A dictionary of :class:`CameraConfig` keyed by camera id.
    """
    raw_ids = merged.get(f"{_ENV_PREFIX}CAMERA_IDS", "")
    cameras: dict[str, CameraConfig] = {}
    for camera_id in (part.strip() for part in raw_ids.split(",") if part.strip()):
        prefix = f"{_ENV_PREFIX}CAMERA_{camera_id.upper()}_"
        kind_name = merged.get(f"{prefix}KIND", "area_scan_2d")
        width = _parse_int(merged.get(f"{prefix}WIDTH", "640"), f"{prefix}WIDTH")
        height = _parse_int(merged.get(f"{prefix}HEIGHT", "480"), f"{prefix}HEIGHT")
        fps = _parse_float(merged.get(f"{prefix}FPS", "30.0"), f"{prefix}FPS")
        device = merged.get(f"{prefix}DEVICE")
        cameras[camera_id] = CameraConfig(
            camera_id=camera_id,
            kind=CameraKind(kind_name),
            width=width,
            height=height,
            fps=fps,
            device=device,
        )
    return cameras


def _parse_int(raw: str, name: str) -> int:
    """Parse ``raw`` as an integer, raising a descriptive error on failure."""
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"Invalid integer for {name}: {raw!r}"
        raise ValueError(msg) from exc


def _parse_float(raw: str, name: str) -> float:
    """Parse ``raw`` as a float, raising a descriptive error on failure."""
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"Invalid number for {name}: {raw!r}"
        raise ValueError(msg) from exc
