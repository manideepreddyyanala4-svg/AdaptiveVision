"""Unit tests for M10 decision fusion and policy."""

from __future__ import annotations

from adaptivevision.common.enums import DefectClass, Severity, Verdict
from adaptivevision.common.result import AnomalyResult, Defect, MetrologyResult
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
