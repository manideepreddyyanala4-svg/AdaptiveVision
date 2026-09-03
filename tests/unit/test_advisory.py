"""Unit tests for the M19 advisory (Ollama root-cause) engine and pipeline."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from adaptivevision.explanation import OllamaAdvisoryEngine
from adaptivevision.explanation import advise, build_evidence
from adaptivevision.common import DefectClass, Severity, Verdict
from adaptivevision.common import AdvisoryError
from adaptivevision.common import AdvisoryEngine
from adaptivevision.common import Defect, InspectionEvidence, RetrievalMatch
from adaptivevision.decision import Decision


def _evidence(*, severity: Severity = Severity.MAJOR) -> InspectionEvidence:
    return InspectionEvidence(
        sample_id="insp-1",
        category="bottle",
        anomaly_score=0.9,
        severity=severity,
        model_ver="patchcore-v1",
        retrieval_matches=(
            RetrievalMatch(
                vector_id=0,
                distance=0.1,
                dataset="mvtec_ad",
                category="bottle",
                defect_type="crack",
            ),
        ),
    )


class _FakeClient:
    """Minimal Ollama-client stand-in for tests."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _valid_payload() -> dict[str, Any]:
    return {
        "defect_classification": "surface crack",
        "confidence_score": 0.8,
        "root_cause_hypothesis": "Likely mold defect based on similar historical cases.",
        "recommended_actions": ["inspect mold", "quarantine batch"],
    }


class _AsAttr:
    """Wraps a payload behind a ``.response`` attribute, like ollama-python."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.response = json.dumps(payload)


def test_generate_report_uses_fallback_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptivevision import explanation as ollama_engine

    monkeypatch.setattr(ollama_engine, "_try_import_ollama", lambda: None)
    engine = OllamaAdvisoryEngine()
    report = engine.generate_report(_evidence(severity=Severity.CRITICAL))
    assert report.is_fallback is True
    assert report.severity == Severity.CRITICAL
    assert report.defect_classification == "crack"  # from the nearest retrieval match
    assert "CRITICAL".lower() in report.root_cause_hypothesis.lower()


def test_generate_report_returns_validated_report_on_success() -> None:
    client = _FakeClient([_AsAttr(_valid_payload())])
    engine = OllamaAdvisoryEngine(client=client)
    report = engine.generate_report(_evidence(severity=Severity.MINOR))
    assert report.is_fallback is False
    assert report.severity == Severity.MINOR
    assert report.defect_classification == "surface crack"
    assert report.confidence_score == pytest.approx(0.8)
    assert report.recommended_actions == ("inspect mold", "quarantine batch")
    assert client.calls[0]["format"] == "json"


def test_generate_report_accepts_dict_style_response() -> None:
    client = _FakeClient([{"response": json.dumps(_valid_payload())}])
    engine = OllamaAdvisoryEngine(client=client)
    report = engine.generate_report(_evidence())
    assert report.is_fallback is False
    assert report.defect_classification == "surface crack"


def test_generate_report_retries_then_falls_back_on_malformed_json() -> None:
    client = _FakeClient([_AsAttr({"not": "the schema"}), _AsAttr({"still": "wrong"})])
    engine = OllamaAdvisoryEngine(client=client, max_retries=1)
    report = engine.generate_report(_evidence())
    assert report.is_fallback is True
    assert len(client.calls) == 2  # initial attempt + 1 retry


def test_generate_report_respects_max_retries_of_zero() -> None:
    client = _FakeClient([_AsAttr({"bad": "payload"})])
    engine = OllamaAdvisoryEngine(client=client, max_retries=0)
    report = engine.generate_report(_evidence())
    assert report.is_fallback is True
    assert len(client.calls) == 1


def test_generate_report_recovers_after_one_bad_attempt() -> None:
    client = _FakeClient([RuntimeError("timeout"), _AsAttr(_valid_payload())])
    engine = OllamaAdvisoryEngine(client=client, max_retries=1)
    report = engine.generate_report(_evidence())
    assert report.is_fallback is False
    assert report.defect_classification == "surface crack"


def test_fallback_report_confidence_is_zero() -> None:
    from adaptivevision.explanation import _fallback_report

    report = _fallback_report(_evidence())
    assert report.confidence_score == 0.0
    assert report.is_fallback is True


def test_fallback_report_without_retrieval_matches() -> None:
    from adaptivevision.explanation import _fallback_report

    evidence = InspectionEvidence(
        sample_id="insp-1",
        category="bottle",
        anomaly_score=0.9,
        severity=Severity.MAJOR,
        model_ver="patchcore-v1",
        retrieval_matches=(),
    )
    report = _fallback_report(evidence)
    assert report.defect_classification == "unknown"
    assert "Most similar historical defect" not in report.root_cause_hypothesis


def test_fallback_report_mentions_heatmap_region_when_present() -> None:
    from adaptivevision.explanation import _fallback_report

    evidence = InspectionEvidence(
        sample_id="insp-1",
        category="bottle",
        anomaly_score=0.9,
        severity=Severity.MAJOR,
        model_ver="patchcore-v1",
        retrieval_matches=(),
        heatmap_region="upper-right",
    )
    report = _fallback_report(evidence)
    assert "upper-right" in report.root_cause_hypothesis


def test_build_prompt_includes_heatmap_region_line_when_present() -> None:
    from adaptivevision.explanation import _build_prompt

    evidence = _evidence()
    with_region = dataclasses.replace(evidence, heatmap_region="lower-center")
    assert "lower-center" in _build_prompt(with_region)
    assert "Heatmap region" not in _build_prompt(evidence)


def test_try_import_ollama_returns_module_when_installed() -> None:
    from adaptivevision.explanation import _try_import_ollama

    # `ollama` is a declared dependency (pyproject's `intelligence` extra) and
    # is expected to be installed wherever this test suite runs.
    module = _try_import_ollama()
    assert module is not None
    assert module.__name__ == "ollama"


def test_try_import_ollama_returns_none_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from adaptivevision import explanation as ollama_engine

    real_import_module = importlib.import_module

    def _raise(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ollama":
            raise ImportError(name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(ollama_engine.importlib, "import_module", _raise)
    assert ollama_engine._try_import_ollama() is None


# --- pipeline ---------------------------------------------------------


def test_build_evidence_uses_highest_severity_among_defects() -> None:
    decision = Decision(
        verdict=Verdict.FAIL,
        defects=(
            Defect(defect_class=DefectClass.ANOMALY, severity=Severity.MINOR),
            Defect(defect_class=DefectClass.ANOMALY, severity=Severity.CRITICAL),
            Defect(defect_class=DefectClass.SCRATCH, severity=Severity.MAJOR),
        ),
    )
    evidence = build_evidence(
        inspection_id="insp-1",
        category="bottle",
        anomaly_score=0.95,
        model_ver="patchcore-v1",
        decision=decision,
    )
    assert evidence.severity == Severity.CRITICAL


def test_build_evidence_defaults_to_info_with_no_defects() -> None:
    decision = Decision(verdict=Verdict.PASS, defects=())
    evidence = build_evidence(
        inspection_id="insp-2",
        category="bottle",
        anomaly_score=0.1,
        model_ver="patchcore-v1",
        decision=decision,
    )
    assert evidence.severity == Severity.INFO


def test_advise_returns_none_when_advisory_disabled() -> None:
    assert advise(_evidence(), advisory=None) is None


def test_advise_returns_report_when_severity_matches() -> None:
    client = _FakeClient([_AsAttr(_valid_payload())])
    engine = OllamaAdvisoryEngine(client=client)
    evidence = _evidence(severity=Severity.MAJOR)
    report = advise(evidence, advisory=engine)
    assert report is not None
    assert report.severity == Severity.MAJOR


class _MisbehavingEngine(AdvisoryEngine):
    """An advisory engine that (incorrectly) changes the severity."""

    def generate_report(self, evidence: InspectionEvidence) -> Any:
        from adaptivevision.common import AdvisoryReport

        return AdvisoryReport(
            defect_classification="x",
            severity=Severity.INFO,  # wrong on purpose
            confidence_score=0.5,
            root_cause_hypothesis="h",
        )


def test_advise_raises_if_engine_overrides_severity() -> None:
    evidence = _evidence(severity=Severity.CRITICAL)
    with pytest.raises(AdvisoryError, match="must never override"):
        advise(evidence, advisory=_MisbehavingEngine())
