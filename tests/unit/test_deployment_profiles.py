"""Unit tests for the M19 deployment-profile loader and Pareto recommender."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptivevision.deployment import (
    DeploymentProfile,
    explain_recommendation,
    feasible_profiles,
    load_deployment_profiles,
    pareto_frontier,
    recommend,
)


def _profile(
    model: str,
    *,
    auroc: float | None,
    p95: float | None,
    params: float | None = 10.0,
    dataset: str = "mvtec_ad",
) -> DeploymentProfile:
    return DeploymentProfile(
        model=model,
        family="memory_bank",
        backbone="resnet50",
        config=f"{model}-cfg",
        dataset=dataset,
        n_seeds=3,
        benchmark_version="2026-08-26",
        validated_at="2026-08-26T00:00:00+00:00",
        image_auroc=auroc,
        p95_latency_ms=p95,
        model_params_millions=params,
    )


def test_to_dict_from_dict_round_trip() -> None:
    profile = _profile("patchcore", auroc=0.99, p95=17.3)
    restored = DeploymentProfile.from_dict(profile.to_dict())
    assert restored == profile


def test_load_deployment_profiles_reads_json_list(tmp_path: Path) -> None:
    path = tmp_path / "deployment_profiles.json"
    path.write_text(json.dumps([_profile("patchcore", auroc=0.99, p95=17.3).to_dict()]))
    profiles = load_deployment_profiles(path)
    assert len(profiles) == 1
    assert profiles[0].model == "patchcore"


def test_load_deployment_profiles_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="JSON list"):
        load_deployment_profiles(path)


def test_pareto_frontier_excludes_dominated_points() -> None:
    a = _profile("a", auroc=0.99, p95=50.0)  # dominated by c
    b = _profile("b", auroc=0.90, p95=10.0)  # non-dominated: much faster
    c = _profile("c", auroc=0.99, p95=20.0)  # dominates a
    frontier = pareto_frontier([a, b, c])
    assert set(frontier) == {b, c}


def test_pareto_frontier_excludes_missing_metrics() -> None:
    complete = _profile("complete", auroc=0.95, p95=20.0)
    missing_auroc = _profile("missing", auroc=None, p95=5.0)
    frontier = pareto_frontier([complete, missing_auroc])
    assert frontier == (complete,)


def test_recommend_picks_highest_auroc_among_feasible() -> None:
    profiles = [
        _profile("slow_accurate", auroc=0.995, p95=90.0),  # infeasible: too slow
        _profile("fast_ok", auroc=0.95, p95=10.0),
        _profile("fast_better", auroc=0.97, p95=15.0),
    ]
    picked = recommend(profiles, max_latency_ms=50.0, min_auroc=0.90)
    assert picked is not None
    assert picked.model == "fast_better"


def test_recommend_returns_none_when_nothing_feasible() -> None:
    profiles = [_profile("only", auroc=0.5, p95=200.0)]
    assert recommend(profiles, max_latency_ms=50.0, min_auroc=0.90) is None


def test_recommend_respects_model_size_constraint() -> None:
    profiles = [
        _profile("small", auroc=0.90, p95=10.0, params=5.0),
        _profile("huge", auroc=0.99, p95=10.0, params=500.0),
    ]
    picked = recommend(profiles, max_latency_ms=50.0, min_auroc=0.80, max_model_size_millions=50.0)
    assert picked is not None
    assert picked.model == "small"


def test_recommend_is_deterministic_across_input_order() -> None:
    profiles = [
        _profile("a", auroc=0.95, p95=10.0),
        _profile("b", auroc=0.95, p95=10.0),
    ]
    first = recommend(profiles, max_latency_ms=50.0, min_auroc=0.5)
    second = recommend(list(reversed(profiles)), max_latency_ms=50.0, min_auroc=0.5)
    assert first == second


def test_feasible_profiles_excludes_missing_model_size_when_constrained() -> None:
    unknown_size = _profile("unknown_size", auroc=0.95, p95=10.0, params=None)
    result = feasible_profiles(
        [unknown_size], max_latency_ms=50.0, min_auroc=0.5, max_model_size_millions=50.0
    )
    assert result == ()


def test_feasible_profiles_filters_by_all_constraints() -> None:
    profiles = [
        _profile("ok", auroc=0.95, p95=10.0, params=5.0),
        _profile("too_slow", auroc=0.95, p95=500.0, params=5.0),
        _profile("too_inaccurate", auroc=0.10, p95=10.0, params=5.0),
        _profile("too_big", auroc=0.95, p95=10.0, params=500.0),
    ]
    result = feasible_profiles(
        profiles, max_latency_ms=50.0, min_auroc=0.5, max_model_size_millions=50.0
    )
    assert [p.model for p in result] == ["ok"]


def test_explain_recommendation_mentions_metrics() -> None:
    profile = _profile("patchcore", auroc=0.991, p95=17.3)
    text = explain_recommendation(profile, max_latency_ms=50.0, min_auroc=0.9, n_feasible=3)
    assert "99.1%" in text
    assert "17.3ms" in text
    assert "patchcore" in text
