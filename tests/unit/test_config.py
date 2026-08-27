"""Unit tests for :mod:`adaptivevision.config`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from adaptivevision.common.enums import CameraKind, ExecutionProvider
from adaptivevision.config import (
    CameraConfig,
    StationConfig,
    load_config,
    load_env_file,
)


def test_station_config_defaults() -> None:
    config = StationConfig(station_id="s1", log_level="INFO")
    assert config.station_id == "s1"
    assert config.log_level == "INFO"
    assert config.cameras == {}
    assert config.default_recipe_id is None
    assert config.execution_provider is ExecutionProvider.CPU
    assert config.cycle_timeout_ms == 5000.0


def test_station_config_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="station_id"):
        StationConfig(station_id="", log_level="INFO")


def test_station_config_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="cycle_timeout_ms"):
        StationConfig(station_id="s", log_level="INFO", cycle_timeout_ms=0.0)


def test_station_config_roundtrip() -> None:
    config = StationConfig(
        station_id="s1",
        log_level="DEBUG",
        cameras={
            "cam0": CameraConfig(
                camera_id="cam0",
                kind=CameraKind.AREA_SCAN_2D,
                width=640,
                height=480,
                fps=30.0,
                device="0",
            )
        },
        default_recipe_id="r1",
        execution_provider=ExecutionProvider.OPENVINO,
        cycle_timeout_ms=1234.5,
        extra={"future": "value"},
    )
    assert StationConfig.from_dict(config.to_dict()) == config


def test_station_config_camera_lookup() -> None:
    config = StationConfig(
        station_id="s",
        log_level="INFO",
        cameras={"cam0": CameraConfig("cam0", CameraKind.AREA_SCAN_2D, 640, 480, 30.0)},
    )
    assert config.camera("cam0").width == 640
    with pytest.raises(KeyError):
        config.camera("nope")


def test_camera_config_validation() -> None:
    with pytest.raises(ValueError, match="width"):
        CameraConfig("c", CameraKind.AREA_SCAN_2D, 0, 480, 30.0)
    with pytest.raises(ValueError, match="height"):
        CameraConfig("c", CameraKind.AREA_SCAN_2D, 640, -1, 30.0)
    with pytest.raises(ValueError, match="fps"):
        CameraConfig("c", CameraKind.AREA_SCAN_2D, 640, 480, 0.0)


def test_station_config_is_frozen() -> None:
    config = StationConfig(station_id="s", log_level="INFO")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.station_id = "other"  # type: ignore[misc]


def test_load_config_defaults(tmp_path: Path) -> None:
    config = load_config(environ={}, env_file=tmp_path / "none.env")
    assert config.station_id == "station-01"
    assert config.log_level == "INFO"
    assert config.cameras == {}
    assert config.execution_provider is ExecutionProvider.CPU


def test_load_config_reads_environment(tmp_path: Path) -> None:
    env = {
        "ADAPTIVEVISION_STATION_ID": "station-9",
        "ADAPTIVEVISION_LOG_LEVEL": "DEBUG",
        "ADAPTIVEVISION_DEFAULT_RECIPE_ID": "widget-a",
        "ADAPTIVEVISION_EXECUTION_PROVIDER": "openvino",
        "ADAPTIVEVISION_CYCLE_TIMEOUT_MS": "2500.0",
    }
    config = load_config(environ=env, env_file=tmp_path / "none.env")
    assert config.station_id == "station-9"
    assert config.log_level == "DEBUG"
    assert config.default_recipe_id == "widget-a"
    assert config.execution_provider is ExecutionProvider.OPENVINO
    assert config.cycle_timeout_ms == 2500.0


def test_load_config_parses_cameras(tmp_path: Path) -> None:
    env = {
        "ADAPTIVEVISION_CAMERA_IDS": "cam0,cam1",
        "ADAPTIVEVISION_CAMERA_CAM0_KIND": "area_scan_2d",
        "ADAPTIVEVISION_CAMERA_CAM0_WIDTH": "1280",
        "ADAPTIVEVISION_CAMERA_CAM0_HEIGHT": "720",
        "ADAPTIVEVISION_CAMERA_CAM0_FPS": "60.0",
        "ADAPTIVEVISION_CAMERA_CAM0_DEVICE": "0",
        "ADAPTIVEVISION_CAMERA_CAM1_KIND": "line_scan_2d",
    }
    config = load_config(environ=env, env_file=tmp_path / "none.env")
    assert set(config.cameras) == {"cam0", "cam1"}
    cam0 = config.cameras["cam0"]
    assert cam0.kind is CameraKind.AREA_SCAN_2D
    assert cam0.width == 1280
    assert cam0.height == 720
    assert cam0.fps == 60.0
    assert cam0.device == "0"
    assert config.cameras["cam1"].kind is CameraKind.LINE_SCAN_2D


def test_load_config_preserves_unknown_vars_in_extra(tmp_path: Path) -> None:
    env = {"ADAPTIVEVISION_FUTURE_SETTING": "hello"}
    config = load_config(environ=env, env_file=tmp_path / "none.env")
    assert config.extra == {"FUTURE_SETTING": "hello"}


def test_load_config_rejects_invalid_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_config(
            environ={"ADAPTIVEVISION_EXECUTION_PROVIDER": "bogus"},
            env_file=tmp_path / "none.env",
        )


def test_load_config_rejects_invalid_number(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CYCLE_TIMEOUT_MS"):
        load_config(
            environ={"ADAPTIVEVISION_CYCLE_TIMEOUT_MS": "abc"},
            env_file=tmp_path / "none.env",
        )


def test_load_env_file_parses_and_respects_precedence(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "ADAPTIVEVISION_STATION_ID=from-file\n"
        "ADAPTIVEVISION_LOG_LEVEL='DEBUG'\n"
        "ALREADY_SET=ignored\n"
        "\n"
        "MALFORMED_LINE\n",
        encoding="utf-8",
    )
    parsed = load_env_file(env_file, environ={"ALREADY_SET": "keep"})
    assert parsed["ADAPTIVEVISION_STATION_ID"] == "from-file"
    assert parsed["ADAPTIVEVISION_LOG_LEVEL"] == "DEBUG"
    assert "ALREADY_SET" not in parsed
    assert "MALFORMED_LINE" not in parsed


def test_load_env_file_missing_path(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "nope.env") == {}


def test_load_config_merges_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ADAPTIVEVISION_STATION_ID=from-file\n", encoding="utf-8")
    config = load_config(environ={}, env_file=env_file)
    assert config.station_id == "from-file"


def test_load_config_defaults_to_dot_env_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``env_file`` omitted entirely -- must still read ``.env`` from cwd.

    Regression test: this previously silently did nothing (``env_file`` stayed
    ``None`` and was never defaulted), so a committed ``.env.example`` copied
    to ``.env`` per its own instructions was never actually read by any
    caller that didn't pass ``env_file=`` explicitly, despite the docstring's
    documented contract.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ADAPTIVEVISION_STATION_ID=from-cwd-dotenv\n", encoding="utf-8")
    config = load_config(environ={})
    assert config.station_id == "from-cwd-dotenv"


def test_load_config_tolerates_missing_dot_env_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config(environ={})
    assert config.station_id == "station-01"
