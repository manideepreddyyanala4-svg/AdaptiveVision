"""Unit tests for config.py: station settings, AOI settings, recipes."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from adaptivevision.common import (
    ROI,
    CameraKind,
    ExecutionProvider,
    MeasurementSpec,
    RecipeError,
    Severity,
    Tolerance,
)
from adaptivevision.config import (
    AoiConfig,
    CameraConfig,
    DecisionPolicy,
    DriftSettings,
    JsonRecipeStore,
    KpiSettings,
    MetrologySettings,
    Recipe,
    StationConfig,
    load_aoi_config,
    load_config,
    load_env_file,
    validate_inspectors,
)

# -----------------------------------------------------------------------------
# Station configuration and env loading
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# AOI settings (metrology/drift/KPI)
# -----------------------------------------------------------------------------

def test_load_aoi_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    config = load_aoi_config(tmp_path / "does-not-exist.yaml")
    assert config == AoiConfig()
    assert config.metrology == MetrologySettings()
    assert config.drift == DriftSettings()
    assert config.kpi == KpiSettings()


def test_load_aoi_config_empty_file_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_aoi_config(path) == AoiConfig()


def test_load_aoi_config_reads_all_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
metrology:
  pixel_to_micron: 2.5
  min_area_px2: 9
  threshold_percentile: 95.0
drift:
  window_size: 50
  p_value_threshold: 0.05
kpi:
  target_escape_rate: 0.002
""",
        encoding="utf-8",
    )

    config = load_aoi_config(path)

    assert config.metrology == MetrologySettings(
        pixel_to_micron=2.5, min_area_px2=9, threshold_percentile=95.0
    )
    assert config.drift == DriftSettings(window_size=50, p_value_threshold=0.05)
    assert config.kpi == KpiSettings(target_escape_rate=0.002)


def test_load_aoi_config_partial_file_fills_in_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("metrology:\n  pixel_to_micron: 3.0\n", encoding="utf-8")

    config = load_aoi_config(path)

    assert config.metrology.pixel_to_micron == 3.0
    assert config.metrology.min_area_px2 == MetrologySettings().min_area_px2
    # Regression check: an absent threshold_percentile key must fall back to
    # the field's own default, not get silently hard-coded to None (a real
    # bug this test caught when that default changed from None to 99.5).
    assert config.metrology.threshold_percentile == MetrologySettings().threshold_percentile
    assert config.drift == DriftSettings()
    assert config.kpi == KpiSettings()


def test_load_aoi_config_null_threshold_percentile_stays_none(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("metrology:\n  threshold_percentile: null\n", encoding="utf-8")

    config = load_aoi_config(path)

    assert config.metrology.threshold_percentile is None


def test_load_aoi_config_default_path_reads_real_repo_config() -> None:
    # configs/config.yaml ships in the repo; this is the one test that
    # exercises the real, no-argument default path end to end.
    config = load_aoi_config()
    assert config.metrology.pixel_to_micron > 0
    assert 0.0 < config.kpi.target_escape_rate < 1.0


# -----------------------------------------------------------------------------
# Recipe model and JSON store
# -----------------------------------------------------------------------------

def _spec(name: str = "width") -> MeasurementSpec:
    return MeasurementSpec(
        name=name,
        nominal=10.0,
        tolerance=Tolerance(minus=0.1, plus=0.1),
        unit="mm",
    )


def _recipe() -> Recipe:
    return Recipe(
        recipe_id="widget-a",
        version="1.0",
        rois=(ROI(label="pad", x=0.0, y=0.0, width=4.0, height=4.0),),
        measurement_specs=(_spec(),),
        inspectors=("metrology",),
        decision=DecisionPolicy(anomaly_threshold=0.7, max_defects=2),
        product_name="Widget A",
    )


def test_recipe_roundtrip() -> None:
    recipe = _recipe()
    assert Recipe.from_dict(recipe.to_dict()) == recipe


def test_recipe_rejects_empty_id() -> None:
    with pytest.raises(RecipeError, match="recipe_id"):
        Recipe(recipe_id="", version="1.0")


def test_recipe_rejects_empty_version() -> None:
    with pytest.raises(RecipeError, match="version"):
        Recipe(recipe_id="r", version="")


def test_recipe_is_frozen() -> None:
    recipe = _recipe()
    with pytest.raises(dataclasses.FrozenInstanceError):
        recipe.version = "2.0"  # type: ignore[misc]


def test_decision_policy_defaults() -> None:
    policy = DecisionPolicy()
    assert policy.anomaly_threshold == 0.5
    assert policy.review_on_anomaly is False
    assert policy.max_defects == 0
    assert policy.fail_severity is Severity.MAJOR


def test_decision_policy_validation() -> None:
    with pytest.raises(RecipeError, match="anomaly_threshold"):
        DecisionPolicy(anomaly_threshold=1.5)
    with pytest.raises(RecipeError, match="max_defects"):
        DecisionPolicy(max_defects=-1)


def test_decision_policy_roundtrip() -> None:
    policy = DecisionPolicy(
        anomaly_threshold=0.8, review_on_anomaly=True, max_defects=3
    )
    assert DecisionPolicy.from_dict(policy.to_dict()) == policy


def test_validate_inspectors_deduplicates_and_preserves_order() -> None:
    registry = frozenset({"metrology", "anomaly"})
    assert validate_inspectors(("metrology", "anomaly", "metrology"), registry) == (
        "metrology",
        "anomaly",
    )


def test_validate_inspectors_rejects_unknown() -> None:
    with pytest.raises(RecipeError, match="Unknown inspector"):
        validate_inspectors(("metrology", "bogus"), frozenset({"metrology"}))


def test_json_store_save_load_roundtrip(tmp_path: Path) -> None:
    store = JsonRecipeStore(tmp_path)
    recipe = _recipe()
    store.save(recipe)
    assert store.load("widget-a") == recipe


def test_json_store_list_ids(tmp_path: Path) -> None:
    store = JsonRecipeStore(tmp_path)
    store.save(_recipe())
    store.save(Recipe(recipe_id="widget-b", version="1.0"))
    assert store.list_ids() == ("widget-a", "widget-b")


def test_json_store_list_ids_empty_when_missing_dir(tmp_path: Path) -> None:
    store = JsonRecipeStore(tmp_path / "nope")
    assert store.list_ids() == ()


def test_json_store_load_missing_raises(tmp_path: Path) -> None:
    store = JsonRecipeStore(tmp_path)
    with pytest.raises(RecipeError, match="not found"):
        store.load("missing")


def test_json_store_load_invalid_json_raises(tmp_path: Path) -> None:
    store = JsonRecipeStore(tmp_path)
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RecipeError, match="Failed to read"):
        store.load("bad")


def test_json_store_load_id_mismatch_raises(tmp_path: Path) -> None:
    store = JsonRecipeStore(tmp_path)
    (tmp_path / "file-a.json").write_text(
        '{"recipe_id": "file-b", "version": "1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(RecipeError, match="mismatch"):
        store.load("file-a")


def test_json_store_is_recipe_store(tmp_path: Path) -> None:
    from adaptivevision.common import RecipeStore

    assert isinstance(JsonRecipeStore(tmp_path), RecipeStore)
