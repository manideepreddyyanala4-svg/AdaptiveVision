"""Unit tests for explanation.py: LLM advisory pipeline and FAISS retrieval."""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adaptivevision.common import (
    AdvisoryEngine,
    AdvisoryError,
    Defect,
    DefectClass,
    InspectionEvidence,
    RetrievalError,
    RetrievalMatch,
    Severity,
    Verdict,
)
from adaptivevision.decision import Decision
from adaptivevision.explanation import (
    FaissRetrievalIndex,
    OllamaAdvisoryEngine,
    advise,
    build_evidence,
)

# -----------------------------------------------------------------------------
# Advisory pipeline and Ollama engine
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# FAISS retrieval index
# -----------------------------------------------------------------------------

class _FakeIndex:
    """A brute-force stand-in for a FAISS flat index."""

    def __init__(self, dim: int, *, use_inner_product: bool, fail: bool = False) -> None:
        self.dim = dim
        self.use_inner_product = use_inner_product
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.fail = fail

    def add(self, vectors: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        if self.fail:
            msg = "simulated FAISS add failure"
            raise RuntimeError(msg)
        self.vectors = np.vstack([self.vectors, vectors])

    def search(
        self, query: np.ndarray[Any, np.dtype[np.float32]], k: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self.fail:
            msg = "simulated FAISS search failure"
            raise RuntimeError(msg)
        n = self.vectors.shape[0]
        if n == 0:
            return (
                np.full((1, k), -1.0, dtype=np.float32),
                np.full((1, k), -1, dtype=np.int64),
            )
        if self.use_inner_product:
            scores = self.vectors @ query[0]
            order = np.argsort(-scores)
        else:
            scores = np.linalg.norm(self.vectors - query[0], axis=1)
            order = np.argsort(scores)
        top = order[:k]
        padded_idx = np.full(k, -1, dtype=np.int64)
        padded_dist = np.full(k, -1.0, dtype=np.float32)
        padded_idx[: len(top)] = top
        padded_dist[: len(top)] = scores[top]
        return padded_dist.reshape(1, -1), padded_idx.reshape(1, -1)


class _FakeFaiss:
    """A minimal FAISS-module stand-in, used by tests."""

    def __init__(self, *, fail_index: bool = False, fail_io: bool = False) -> None:
        self._fail_index = fail_index
        self._fail_io = fail_io

    def IndexFlatL2(self, dim: int) -> _FakeIndex:  # noqa: N802
        return _FakeIndex(dim, use_inner_product=False, fail=self._fail_index)

    def IndexFlatIP(self, dim: int) -> _FakeIndex:  # noqa: N802
        return _FakeIndex(dim, use_inner_product=True, fail=self._fail_index)

    def write_index(self, index: _FakeIndex, path: str) -> None:
        if self._fail_io:
            msg = "simulated FAISS write_index failure"
            raise RuntimeError(msg)
        Path(path).write_bytes(pickle.dumps(index))

    def read_index(self, path: str) -> _FakeIndex:
        if self._fail_io:
            msg = "simulated FAISS read_index failure"
            raise RuntimeError(msg)
        return pickle.loads(Path(path).read_bytes())


def _index(**kwargs: Any) -> FaissRetrievalIndex:
    return FaissRetrievalIndex(4, faiss_module=_FakeFaiss(), **kwargs)


def _rows(*values: list[float]) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.array(values, dtype=np.float32)


def _meta(n: int) -> list[dict[str, str]]:
    return [
        {"dataset": "mvtec_ad", "category": "bottle", "defect_type": f"defect-{i}"}
        for i in range(n)
    ]


def test_add_assigns_sequential_ids_and_returns_them() -> None:
    index = _index()
    ids = index.add(_rows([1, 0, 0, 0], [0, 1, 0, 0]), _meta(2))
    assert ids == (0, 1)
    more_ids = index.add(_rows([0, 0, 1, 0]), _meta(1))
    assert more_ids == (2,)


def test_search_returns_nearest_match_with_metadata() -> None:
    index = _index(metric="l2")
    index.add(
        _rows([0, 0, 0, 0], [10, 10, 10, 10], [1, 0, 0, 0]),
        [
            {"dataset": "mvtec_ad", "category": "bottle", "defect_type": "crack"},
            {"dataset": "visa", "category": "capsules", "defect_type": "poke"},
            {"dataset": "kolektorsdd2", "category": "surface", "defect_type": "blob"},
        ],
    )
    results = index.search(np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].vector_id == 0
    assert results[0].dataset == "mvtec_ad"
    assert results[0].defect_type == "crack"


def test_search_omits_padding_when_fewer_than_top_k_results() -> None:
    index = _index()
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    results = index.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
    assert len(results) == 1


def test_cosine_metric_normalizes_before_add_and_search() -> None:
    index = _index(metric="cosine")
    # Two vectors that are positive scalar multiples of each other should be
    # indistinguishable after L2 normalization.
    index.add(_rows([1, 0, 0, 0], [50, 0, 0, 0]), _meta(2))
    results = index.search(np.array([2, 0, 0, 0], dtype=np.float32), top_k=2)
    assert {r.vector_id for r in results} == {0, 1}
    assert results[0].distance == pytest.approx(results[1].distance, abs=1e-5)


def test_add_rejects_dimension_mismatch() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match="dimension"):
        index.add(_rows([1, 2, 3]), _meta(1))


def test_add_rejects_non_finite_values() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match=r"NaN|infinite"):
        index.add(_rows([1, float("nan"), 0, 0]), _meta(1))


def test_add_rejects_metadata_length_mismatch() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match="metadata"):
        index.add(_rows([1, 0, 0, 0], [0, 1, 0, 0]), _meta(1))


def test_add_accepts_empty_input() -> None:
    index = _index()
    assert index.add(np.empty((0, 4), dtype=np.float32), []) == ()


def test_search_rejects_dimension_mismatch() -> None:
    index = _index()
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    with pytest.raises(RetrievalError, match="dimension"):
        index.search(np.array([1, 0, 0], dtype=np.float32))


def test_unsupported_index_type_rejected() -> None:
    with pytest.raises(RetrievalError, match="index_type"):
        FaissRetrievalIndex(4, index_type="ivf", faiss_module=_FakeFaiss())


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = _index(embedding_model="dinov2", embedding_version="v1")
    index.add(_rows([1, 0, 0, 0], [0, 1, 0, 0]), _meta(2))
    path = tmp_path / "index.faiss"
    index.save(path)

    reloaded = FaissRetrievalIndex(
        4, faiss_module=_FakeFaiss(), embedding_model="dinov2", embedding_version="v1"
    )
    reloaded.load(path)
    results = reloaded.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
    assert results[0].vector_id == 0
    assert results[0].dataset == "mvtec_ad"


def test_load_rejects_incompatible_embedding_configuration(tmp_path: Path) -> None:
    index = _index(embedding_model="dinov2", embedding_version="v1")
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    path = tmp_path / "index.faiss"
    index.save(path)

    mismatched = FaissRetrievalIndex(
        4, faiss_module=_FakeFaiss(), embedding_model="resnet50", embedding_version="v1"
    )
    with pytest.raises(RetrievalError, match="incompatible"):
        mismatched.load(path)


def test_import_faiss_raises_retrieval_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from adaptivevision import explanation as faiss_index

    real_import_module = importlib.import_module

    def _raise(name: str, *args: object, **kwargs: object) -> object:
        if name == "faiss":
            raise ImportError(name)
        return real_import_module(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(faiss_index.importlib, "import_module", _raise)
    with pytest.raises(RetrievalError, match="faiss is not installed"):
        faiss_index._import_faiss()


def test_metadata_info_exposes_index_configuration() -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(), metric="l2", embedding_model="dinov2")
    info = index.metadata_info
    assert info.dim == 4
    assert info.metric == "l2"
    assert info.embedding_model == "dinov2"


def test_add_wraps_underlying_faiss_failure() -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(fail_index=True))
    with pytest.raises(RetrievalError, match="Failed to add"):
        index.add(_rows([1, 0, 0, 0]), _meta(1))


def test_search_wraps_underlying_faiss_failure() -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(fail_index=True))
    with pytest.raises(RetrievalError, match="FAISS search failed"):
        index.search(np.array([1, 0, 0, 0], dtype=np.float32))


def test_save_wraps_underlying_io_failure(tmp_path: Path) -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(fail_io=True))
    with pytest.raises(RetrievalError, match="Failed to save"):
        index.save(tmp_path / "index.faiss")


def test_load_wraps_generic_failure_distinctly_from_incompatibility(
    tmp_path: Path,
) -> None:
    # A missing/corrupt sidecar is a different failure mode than the
    # "incompatible configuration" RetrievalError raised deliberately -
    # both must surface as RetrievalError, but via the generic except path.
    index = _index()
    with pytest.raises(RetrievalError, match="Failed to load"):
        index.load(tmp_path / "does-not-exist.faiss")


def test_add_rejects_wrong_number_of_dimensions() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match="Expected a 2D array"):
        index.add(np.array([1, 0, 0, 0], dtype=np.float32), _meta(1))


def test_search_rejects_wrong_number_of_dimensions() -> None:
    index = _index()
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    with pytest.raises(RetrievalError, match="Expected a 1D array"):
        index.search(np.array([[1, 0, 0, 0]], dtype=np.float32).reshape(1, 1, 4))
