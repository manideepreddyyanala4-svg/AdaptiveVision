"""Unit tests for :mod:`adaptivevision.recipe`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from adaptivevision.common.enums import Severity
from adaptivevision.common.errors import RecipeError
from adaptivevision.common.types import ROI, MeasurementSpec, Tolerance
from adaptivevision.recipe import (
    DecisionPolicy,
    JsonRecipeStore,
    Recipe,
    validate_inspectors,
)


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
    policy = DecisionPolicy(anomaly_threshold=0.8, review_on_anomaly=True, max_defects=3)
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
    from adaptivevision.common.interfaces import RecipeStore

    assert isinstance(JsonRecipeStore(tmp_path), RecipeStore)
