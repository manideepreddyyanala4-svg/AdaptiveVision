"""Unit tests for M10 decision fusion and policy."""

from __future__ import annotations

from adaptivevision.common import (
    AnomalyResult,
    Defect,
    DefectClass,
    MetrologyResult,
    Severity,
    Verdict,
)
from adaptivevision.config import DecisionPolicy as RecipeDecisionPolicy
from adaptivevision.config import Recipe
from adaptivevision.decision import DecisionPolicy


def _defect(severity: Severity) -> Defect:
    return Defect(
        defect_class=DefectClass.ANOMALY,
        severity=severity,
        score=0.9,
        description="test defect",
    )


def test_empty_partials_pass() -> None:
    policy = DecisionPolicy()
    decision = policy.decide([])
    assert decision.verdict == Verdict.PASS
    assert decision.defects == ()


def test_major_defect_fails() -> None:
    policy = DecisionPolicy()
    partial = MetrologyResult(defects=(_defect(Severity.MAJOR),))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.FAIL


def test_critical_defect_fails() -> None:
    policy = DecisionPolicy()
    partial = MetrologyResult(defects=(_defect(Severity.CRITICAL),))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.FAIL


def test_minor_defect_reviews() -> None:
    policy = DecisionPolicy()
    partial = MetrologyResult(defects=(_defect(Severity.MINOR),))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.REVIEW


def test_info_defect_reviews() -> None:
    policy = DecisionPolicy()
    partial = MetrologyResult(defects=(_defect(Severity.INFO),))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.REVIEW


def test_anomaly_review_band_routes_to_review() -> None:
    policy = DecisionPolicy(anomaly_review_threshold=0.7)
    partial = AnomalyResult(score=0.75, threshold=0.9, is_anomalous=False)
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.REVIEW


def test_anomaly_below_review_band_passes() -> None:
    policy = DecisionPolicy(anomaly_review_threshold=0.7)
    partial = AnomalyResult(score=0.5, threshold=0.9, is_anomalous=False)
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.PASS


def test_anomalous_part_fails_via_defect() -> None:
    policy = DecisionPolicy(anomaly_review_threshold=0.7)
    partial = AnomalyResult(
        score=0.95,
        threshold=0.9,
        is_anomalous=True,
        defects=(_defect(Severity.MAJOR),),
    )
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.FAIL


def test_defects_fused_across_partials() -> None:
    policy = DecisionPolicy()
    metrology = MetrologyResult(defects=(_defect(Severity.MAJOR),))
    anomaly = AnomalyResult(
        score=0.95,
        threshold=0.9,
        is_anomalous=True,
        defects=(_defect(Severity.CRITICAL),),
    )
    decision = policy.decide([metrology, anomaly])
    assert decision.verdict == Verdict.FAIL
    assert len(decision.defects) == 2


def test_max_defects_forces_fail_even_below_fail_severity() -> None:
    policy = DecisionPolicy(max_defects=1)
    partial = MetrologyResult(defects=(_defect(Severity.MINOR), _defect(Severity.MINOR)))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.FAIL


def test_max_defects_not_exceeded_still_reviews() -> None:
    policy = DecisionPolicy(max_defects=2)
    partial = MetrologyResult(defects=(_defect(Severity.MINOR),))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.REVIEW


def test_configurable_fail_severity_downgrades_major_to_review() -> None:
    policy = DecisionPolicy(fail_severity=Severity.CRITICAL)
    partial = MetrologyResult(defects=(_defect(Severity.MAJOR),))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.REVIEW


def test_configurable_fail_severity_escalates_minor_to_fail() -> None:
    policy = DecisionPolicy(fail_severity=Severity.MINOR)
    partial = MetrologyResult(defects=(_defect(Severity.MINOR),))
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.FAIL


def test_from_recipe_translates_declared_policy() -> None:
    recipe = Recipe(
        recipe_id="widget-a",
        version="1",
        decision=RecipeDecisionPolicy(
            anomaly_threshold=0.6,
            max_defects=3,
            fail_severity=Severity.CRITICAL,
        ),
    )
    policy = DecisionPolicy.from_recipe(recipe)

    # max_defects=3 not exceeded by 3 MAJOR defects, and fail_severity is
    # CRITICAL so MAJOR alone must not force FAIL -- both fields round-tripped.
    partial = MetrologyResult(
        defects=(
            _defect(Severity.MAJOR),
            _defect(Severity.MAJOR),
            _defect(Severity.MAJOR),
        )
    )
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.REVIEW


def test_from_recipe_zero_max_defects_means_no_cap() -> None:
    recipe = Recipe(recipe_id="widget-a", version="1")  # default max_defects=0
    policy = DecisionPolicy.from_recipe(recipe)

    partial = MetrologyResult(defects=(_defect(Severity.MINOR),) * 5)
    decision = policy.decide([partial])
    assert decision.verdict == Verdict.REVIEW
