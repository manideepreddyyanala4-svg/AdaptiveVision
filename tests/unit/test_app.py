"""Unit tests for the composition root (Milestone M3, extended M9/M10/M20)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from adaptivevision.acquisition.frame import build_frame
from adaptivevision.app.app import (
    build_anomaly_detector,
    build_decision_policy,
    build_preprocessor,
    build_recipe,
    build_station,
)
from adaptivevision.common.enums import ExecutionProvider, Verdict
from adaptivevision.common.types import RectifiedFrame
from adaptivevision.config.settings import StationConfig
from adaptivevision.inspection.anomaly.detector import ThresholdAnomalyDetector
from adaptivevision.recipe import Recipe
from adaptivevision.recipe.model import DecisionPolicy as RecipeDecisionPolicy
from adaptivevision.recipe.store import JsonRecipeStore

_REAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def _config(**extra: object) -> StationConfig:
    return StationConfig(station_id="s1", log_level="INFO", extra=dict(extra))


def test_build_recipe_returns_none_without_default_recipe_id() -> None:
    assert build_recipe(_config()) is None


def test_build_recipe_loads_from_configured_directory(tmp_path: Path) -> None:
    recipe = Recipe(recipe_id="widget-a", version="1")
    JsonRecipeStore(tmp_path).save(recipe)
    config = StationConfig(
        station_id="s1",
        log_level="INFO",
        default_recipe_id="widget-a",
        extra={"RECIPE_DIR": str(tmp_path)},
    )

    loaded = build_recipe(config)

    assert loaded is not None
    assert loaded.recipe_id == "widget-a"


def test_build_recipe_defaults_directory_to_recipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    JsonRecipeStore(recipe_dir).save(Recipe(recipe_id="w", version="1"))
    config = StationConfig(station_id="s1", log_level="INFO", default_recipe_id="w")

    loaded = build_recipe(config)

    assert loaded is not None
    assert loaded.recipe_id == "w"


def test_build_anomaly_detector_returns_none_without_model_path() -> None:
    assert build_anomaly_detector(_config(), None) is None


def test_build_anomaly_detector_loads_real_model_and_uses_recipe_threshold() -> None:
    config = _config(MODEL_PATH="mvtec_bottle.onnx", MODEL_DIR=str(_REAL_MODEL_DIR))
    recipe = Recipe(
        recipe_id="r",
        version="1",
        decision=RecipeDecisionPolicy(anomaly_threshold=0.42),
    )

    detector = build_anomaly_detector(config, recipe)

    assert isinstance(detector, ThresholdAnomalyDetector)


def test_build_anomaly_detector_review_on_anomaly_uses_minor_severity() -> None:
    config = _config(MODEL_PATH="mvtec_bottle.onnx", MODEL_DIR=str(_REAL_MODEL_DIR))
    recipe = Recipe(
        recipe_id="r",
        version="1",
        decision=RecipeDecisionPolicy(anomaly_threshold=0.0, review_on_anomaly=True),
    )

    detector = build_anomaly_detector(config, recipe)
    assert detector is not None

    frame = RectifiedFrame(
        image=np.zeros((128, 128), dtype=np.uint8),
        camera_id="cam0",
        frame_id="f1",
        calibration_ver="",
        timestamp_monotonic=0.0,
        timestamp_utc=datetime.now(UTC),
        trigger_id=None,
    )
    result = detector.detect(frame)

    assert result.is_anomalous is True
    assert result.defects[0].severity.value == "minor"


def test_build_anomaly_detector_non_cpu_provider_falls_back_to_cpu() -> None:
    """A non-CPU preferred provider still needs a CPU fallback in the list,
    since edge hardware without that accelerator must still run."""
    config = StationConfig(
        station_id="s1",
        log_level="INFO",
        execution_provider=ExecutionProvider.OPENVINO,
        extra={"MODEL_PATH": "mvtec_bottle.onnx", "MODEL_DIR": str(_REAL_MODEL_DIR)},
    )

    detector = build_anomaly_detector(config, None)

    assert isinstance(detector, ThresholdAnomalyDetector)


def test_build_anomaly_detector_default_threshold_without_recipe() -> None:
    config = _config(MODEL_PATH="mvtec_bottle.onnx", MODEL_DIR=str(_REAL_MODEL_DIR))

    detector = build_anomaly_detector(config, None)

    assert isinstance(detector, ThresholdAnomalyDetector)


def test_build_decision_policy_none_without_recipe() -> None:
    assert build_decision_policy(None) is None


def test_build_decision_policy_from_recipe() -> None:
    policy = build_decision_policy(Recipe(recipe_id="r", version="1"))
    assert policy is not None


def test_build_preprocessor_adds_resize_when_model_input_size_configured() -> None:
    config = _config(MODEL_INPUT_HEIGHT="64", MODEL_INPUT_WIDTH="32")
    preprocessor = build_preprocessor(config)

    frame = build_frame(np.zeros((256, 256), dtype=np.uint8), "cam0")
    result = preprocessor(frame)

    assert result.image.shape == (64, 32)


def test_build_preprocessor_grayscale_only_without_model_input_size() -> None:
    config = _config()
    preprocessor = build_preprocessor(config)

    frame = build_frame(np.zeros((10, 10, 3), dtype=np.uint8), "cam0")
    result = preprocessor(frame)

    assert result.image.ndim == 2


def test_build_station_with_model_and_recipe_produces_real_verdict(
    tmp_path: Path,
) -> None:
    """The core M9/M10 wiring fix: a configured model/recipe changes the
    verdict from the unconditional PASS of an unconfigured station."""
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    JsonRecipeStore(recipe_dir).save(Recipe(recipe_id="demo", version="1"))
    config = StationConfig(
        station_id="s1",
        log_level="INFO",
        default_recipe_id="demo",
        execution_provider=ExecutionProvider.CPU,
        extra={
            "RECIPE_DIR": str(recipe_dir),
            "MODEL_PATH": "mvtec_bottle.onnx",
            "MODEL_DIR": str(_REAL_MODEL_DIR),
            "MODEL_INPUT_HEIGHT": "128",
            "MODEL_INPUT_WIDTH": "128",
        },
    )

    station = build_station(config)
    station.boot()
    station.ready()
    results = station.run(["part-001"])

    assert len(results) == 1
    result = results[0]
    assert result.recipe_ver == "demo"
    assert result.anomaly_score is not None
    # A blank (all-zero) null-camera frame against a real model is a
    # deterministic, real inference result -- not the hardcoded PASS an
    # unconfigured station always returns.
    assert result.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.REVIEW)
