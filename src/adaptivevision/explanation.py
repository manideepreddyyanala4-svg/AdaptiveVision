"""Explain stage: say why, in plain language.

Two things that work together: a local LLM (Ollama) that writes a grounded
root-cause explanation for a deterministic decision it never gets to
override, and FAISS-backed historical-defect retrieval that surfaces similar
past defects as grounding evidence for that explanation.

The severity is already decided (``decision.py``) before the LLM ever runs --
:func:`build_evidence` only *reads* an already-final
:class:`~adaptivevision.decision.Decision`, and :func:`advise` enforces that
whatever an :class:`~adaptivevision.common.AdvisoryEngine` returns still
carries the same severity it was given. A missing ``ollama``/``faiss``
package or an unreachable server both fall through to a deterministic
fallback path rather than erroring -- these are optional advisory services,
not core production dependencies.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from adaptivevision.common import (
    AdvisoryEngine,
    AdvisoryError,
    AdvisoryReport,
    InspectionEvidence,
    RetrievalError,
    RetrievalIndex,
    RetrievalMatch,
    Severity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from adaptivevision.common import Defect, Embedding
    from adaptivevision.decision import Decision

logger = logging.getLogger(__name__)

# =============================================================================
# Advisory pipeline orchestration
#
# Wires the deterministic decision output and the advisory engine together in
# the fixed order the architecture requires.
# =============================================================================

#: Severities ranked lowest to highest, matching the decision policy's rules.
_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.INFO,
    Severity.MINOR,
    Severity.MAJOR,
    Severity.CRITICAL,
)


def build_evidence(
    *,
    inspection_id: str,
    category: str,
    anomaly_score: float | None,
    model_ver: str,
    decision: Decision,
    retrieval_matches: tuple[RetrievalMatch, ...] = (),
    heatmap_region: str | None = None,
) -> InspectionEvidence:
    """Build read-only advisory evidence from an already-final ``decision``.

    Args:
        inspection_id: Identifier of the inspected sample.
        category: Product/part category being inspected.
        anomaly_score: Overall anomaly score, if computed.
        model_ver: Version of the anomaly model that produced the score.
        decision: The final, already-computed decision. Never recomputed.
        retrieval_matches: Historical matches retrieved for this sample.
        heatmap_region: Coarse location of the strongest per-patch anomaly
            signal, if a localization heatmap was computed.

    Returns:
        Evidence carrying the decision's own severity, unchanged.
    """
    return InspectionEvidence(
        sample_id=inspection_id,
        category=category,
        anomaly_score=anomaly_score,
        severity=_max_severity(decision.defects),
        model_ver=model_ver,
        retrieval_matches=retrieval_matches,
        heatmap_region=heatmap_region,
    )


def advise(
    evidence: InspectionEvidence, *, advisory: AdvisoryEngine | None
) -> AdvisoryReport | None:
    """Produce an advisory report for ``evidence``, or ``None`` if disabled.

    Args:
        evidence: Evidence built by :func:`build_evidence`.
        advisory: The advisory engine to use, or ``None`` to skip advisory
            entirely (a fully valid, supported configuration).

    Returns:
        The validated report, or ``None`` if ``advisory`` is ``None``.

    Raises:
        AdvisoryError: If ``advisory`` returns a report whose severity does
            not match ``evidence.severity`` - a bug in that implementation,
            not a normal failure mode, and never expected to trigger in
            practice given :class:`OllamaAdvisoryEngine` always echoes it.
    """
    if advisory is None:
        return None
    report = advisory.generate_report(evidence)
    if report.severity != evidence.severity:
        msg = (
            f"Advisory engine {type(advisory).__name__!r} returned severity "
            f"{report.severity!r} but evidence severity was "
            f"{evidence.severity!r}; advisory engines must never override "
            "the deterministic severity."
        )
        raise AdvisoryError(msg, recoverable=False)
    return report


def _max_severity(defects: Sequence[Defect]) -> Severity:
    """Return the highest-ranked severity among ``defects``, or INFO if empty."""
    if not defects:
        return Severity.INFO
    return max(defects, key=lambda d: _SEVERITY_ORDER.index(d.severity)).severity


# =============================================================================
# Ollama advisory engine
#
# Validates the LLM's raw JSON response with Pydantic (the only place
# Pydantic is used - a raw LLM response is untrusted input) before converting
# it to the stable AdvisoryReport frozen dataclass used everywhere else.
# Deliberately absent from the response schema: a severity field - severity
# is never something the LLM is asked or allowed to set.
# =============================================================================


class RootCauseReportModel(BaseModel):
    """Schema an LLM JSON response must satisfy before being trusted.

    Attributes:
        defect_classification: The model's descriptive classification of the
            defect (a hypothesis, not a ground-truth label).
        confidence_score: Confidence in the hypothesis, constrained to
            ``[0, 1]``.
        root_cause_hypothesis: Explanatory hypothesis grounded in the
            supplied evidence.
        recommended_actions: Suggested next steps.
    """

    defect_classification: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    root_cause_hypothesis: str
    recommended_actions: list[str] = Field(default_factory=list)


#: Default local model. Override via the ``model`` constructor argument.
DEFAULT_MODEL = "qwen2.5:7b"

_SYSTEM_PROMPT = (
    "You are a manufacturing quality-inspection assistant helping a line "
    "operator decide what to do about one inspected part. You are given "
    "deterministic evidence about it. Respond ONLY with a JSON object "
    "matching the requested schema.\n"
    "Rules:\n"
    "- Do not invent evidence beyond what is supplied -- no specific defect "
    "geometry, root cause, or process detail that isn't implied by the "
    "evidence given.\n"
    "- Clearly distinguish your hypothesis from established fact.\n"
    "- The severity has already been determined by a deterministic system "
    "and is not part of your response; do not restate or contradict it.\n"
    "- If you are uncertain, say so in root_cause_hypothesis and lower "
    "confidence_score accordingly.\n"
    "- root_cause_hypothesis should be 2-4 sentences: name the most likely "
    "failure mode given the evidence, reason briefly about why (referencing "
    "the score, the heatmap region, and any historical matches supplied), "
    "and note what would need to be true for an alternative explanation.\n"
    "- recommended_actions should be 2-4 concrete, specific next steps a "
    "line operator or quality engineer could actually do next (not generic "
    "advice like 'investigate further') -- ground each one in the evidence "
    "given, e.g. referencing the flagged region or the matched defect type."
)


class OllamaAdvisoryEngine(AdvisoryEngine):
    """An :class:`~adaptivevision.common.AdvisoryEngine` backed by a local
    Ollama LLM.

    Never raises: a missing ``ollama`` package, an unreachable server, a
    timeout, or a malformed response all fall back to a deterministic report
    derived only from the supplied evidence.

    Args:
        model: Ollama model identifier.
        max_retries: Number of retries after a malformed structured response,
            before falling back.
        client: Optional Ollama-client-like object, used by tests.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        """Initialize the engine without connecting to a server."""
        self._model = model
        self._max_retries = max_retries
        self._client = client

    def generate_report(self, evidence: InspectionEvidence) -> AdvisoryReport:
        """Produce a validated advisory report, falling back if the LLM is unavailable."""
        client = self._client if self._client is not None else _try_import_ollama()
        if client is None:
            logger.info("Ollama unavailable; using deterministic fallback report")
            return _fallback_report(evidence)

        prompt = _build_prompt(evidence)
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                validated = _call_ollama(client, self._model, prompt)
            except Exception as exc:
                logger.warning("Ollama advisory attempt %d/%d failed: %s", attempt, attempts, exc)
                continue
            return _to_domain(validated, evidence)
        return _fallback_report(evidence)


def _call_ollama(client: Any, model: str, prompt: str) -> RootCauseReportModel:
    """Call the Ollama client and validate its response.

    Raises:
        Exception: On any client, JSON-decoding, or validation failure. The
            caller treats this as a single failed attempt, not a fatal error.
    """
    response = client.generate(
        model=model,
        system=_SYSTEM_PROMPT,
        prompt=prompt,
        format="json",
    )
    text = response["response"] if isinstance(response, dict) else response.response
    payload = json.loads(text)
    try:
        return RootCauseReportModel.model_validate(payload)
    except ValidationError as exc:
        msg = f"Ollama response failed schema validation: {exc}"
        raise ValueError(msg) from exc


def _build_prompt(evidence: InspectionEvidence) -> str:
    """Build the evidence prompt sent to the model."""
    matches = (
        "\n".join(
            f"  - {m.defect_type} ({m.dataset}/{m.category}), distance={m.distance:.4f}"
            for m in evidence.retrieval_matches
        )
        or "  (none)"
    )
    region_line = (
        f"Heatmap region (where the per-patch anomaly signal concentrates): "
        f"{evidence.heatmap_region}\n"
        if evidence.heatmap_region
        else ""
    )
    return (
        f"Category: {evidence.category}\n"
        f"Anomaly score: {evidence.anomaly_score}\n"
        f"Deterministic severity: {evidence.severity.value}\n"
        f"Model version: {evidence.model_ver}\n"
        f"{region_line}"
        f"Historical similar defects (nearest first):\n{matches}\n\n"
        'Respond with JSON: {"defect_classification": str, '
        '"confidence_score": float in [0,1], "root_cause_hypothesis": str '
        "(2-4 sentences, reasoned from the evidence above), "
        '"recommended_actions": [str, ...] (2-4 concrete steps)}'
    )


def _to_domain(validated: RootCauseReportModel, evidence: InspectionEvidence) -> AdvisoryReport:
    """Convert a validated LLM response into the stable domain report."""
    return AdvisoryReport(
        defect_classification=validated.defect_classification,
        severity=evidence.severity,
        confidence_score=validated.confidence_score,
        root_cause_hypothesis=validated.root_cause_hypothesis,
        recommended_actions=tuple(validated.recommended_actions),
        is_fallback=False,
    )


def _fallback_report(evidence: InspectionEvidence) -> AdvisoryReport:
    """Build a deterministic report from evidence alone, no LLM involved."""
    nearest = evidence.retrieval_matches[0] if evidence.retrieval_matches else None
    hypothesis = f"No LLM analysis available. Deterministic severity is {evidence.severity.value}."
    if evidence.heatmap_region:
        hypothesis += f" Anomaly signal concentrated in the {evidence.heatmap_region}."
    if nearest is not None:
        hypothesis += (
            f" Most similar historical defect on record: {nearest.defect_type} "
            f"in {nearest.dataset}/{nearest.category}."
        )
    return AdvisoryReport(
        defect_classification=nearest.defect_type if nearest is not None else "unknown",
        severity=evidence.severity,
        confidence_score=0.0,
        root_cause_hypothesis=hypothesis,
        recommended_actions=(),
        is_fallback=True,
    )


def _try_import_ollama() -> Any | None:
    """Import the Ollama client lazily, returning ``None`` if unavailable."""
    try:
        return importlib.import_module("ollama")
    except ImportError:
        return None


# =============================================================================
# FAISS-backed historical-defect retrieval
#
# FAISS stores and searches vectors only; it is never the source of truth for
# business metadata. Each index keeps a small, versioned JSON sidecar
# (IndexMetadata) alongside the binary FAISS file so a saved index records
# the embedding model/version and preprocessing version it was built with,
# and refuses to silently mix incompatible embeddings back in on load.
# =============================================================================

#: FAISS index flavors implemented so far. ``index_type`` is a configuration
#: point for future extension; only "flat" is supported today.
_SUPPORTED_INDEX_TYPES = ("flat",)

_SIDECAR_SUFFIX = ".meta.json"


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    """Configuration and provenance of a saved :class:`FaissRetrievalIndex`.

    Attributes:
        dim: Embedding dimensionality.
        metric: Distance metric the index was built with.
        index_type: FAISS index flavor.
        embedding_model: Identifier of the model that produced the embeddings.
        embedding_version: Version of ``embedding_model``.
        preprocessing_version: Version of the preprocessing applied before
            embedding.
        created_at: UTC timestamp the index was constructed.
    """

    dim: int
    metric: Literal["l2", "ip", "cosine"]
    index_type: str
    embedding_model: str = ""
    embedding_version: str = ""
    preprocessing_version: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "dim": self.dim,
            "metric": self.metric,
            "index_type": self.index_type,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "preprocessing_version": self.preprocessing_version,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            dim=data["dim"],
            metric=data["metric"],
            index_type=data["index_type"],
            embedding_model=data.get("embedding_model", ""),
            embedding_version=data.get("embedding_version", ""),
            preprocessing_version=data.get("preprocessing_version", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def is_compatible_with(self, other: IndexMetadata) -> bool:
        """Return ``True`` if embeddings built under ``other`` may be mixed in."""
        return (
            self.dim == other.dim
            and self.metric == other.metric
            and self.index_type == other.index_type
            and self.embedding_model == other.embedding_model
            and self.embedding_version == other.embedding_version
            and self.preprocessing_version == other.preprocessing_version
        )


class FaissRetrievalIndex(RetrievalIndex):
    """A :class:`~adaptivevision.common.RetrievalIndex` backed by FAISS.

    Vector IDs are assigned sequentially by insertion order and are stable
    for the lifetime of the index (there is no removal).

    Args:
        dim: Embedding dimensionality.
        metric: Distance metric. ``"cosine"`` is implemented as inner-product
            search over L2-normalized vectors.
        index_type: FAISS index flavor. Only ``"flat"`` is supported.
        embedding_model: Identifier of the model producing embeddings, stored
            in the index metadata for compatibility checks on load.
        embedding_version: Version of ``embedding_model``.
        preprocessing_version: Version of the preprocessing applied before
            embedding.
        faiss_module: Optional FAISS-like module, used by tests.

    Raises:
        RetrievalError: If ``index_type`` is not supported.
    """

    def __init__(
        self,
        dim: int,
        *,
        metric: Literal["l2", "ip", "cosine"] = "cosine",
        index_type: str = "flat",
        embedding_model: str = "",
        embedding_version: str = "",
        preprocessing_version: str = "",
        faiss_module: Any | None = None,
    ) -> None:
        """Initialize the index and build the underlying FAISS structure."""
        if index_type not in _SUPPORTED_INDEX_TYPES:
            msg = (
                f"Unsupported FAISS index_type {index_type!r}; "
                f"supported: {_SUPPORTED_INDEX_TYPES}"
            )
            raise RetrievalError(msg)
        self._dim = dim
        self._metric: Literal["l2", "ip", "cosine"] = metric
        self._index_type = index_type
        self._faiss = faiss_module or _import_faiss()
        self._metadata_meta = IndexMetadata(
            dim=dim,
            metric=metric,
            index_type=index_type,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            preprocessing_version=preprocessing_version,
        )
        self._index = self._build_index()
        self._metadata: list[dict[str, Any]] = []

    @property
    def metadata_info(self) -> IndexMetadata:
        """Return the index's configuration/provenance metadata."""
        return self._metadata_meta

    def add(self, embeddings: Embedding, metadata: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        """Add embeddings with associated metadata.

        Raises:
            RetrievalError: On dimension mismatch, non-finite values, or a
                length mismatch between ``embeddings`` and ``metadata``.
        """
        vectors = self._validate_embeddings(embeddings, expected_ndim=2)
        if vectors.shape[0] != len(metadata):
            msg = (
                f"embeddings has {vectors.shape[0]} rows but metadata has "
                f"{len(metadata)} entries"
            )
            raise RetrievalError(msg)
        if vectors.shape[0] == 0:
            return ()
        vectors = self._normalize_if_cosine(vectors)
        start_id = len(self._metadata)
        try:
            self._index.add(vectors)
        except Exception as exc:
            msg = f"Failed to add {vectors.shape[0]} embeddings to the FAISS index: {exc}"
            raise RetrievalError(msg) from exc
        self._metadata.extend(dict(m) for m in metadata)
        return tuple(range(start_id, start_id + vectors.shape[0]))

    def search(self, query: Embedding, top_k: int = 3) -> tuple[RetrievalMatch, ...]:
        """Return the ``top_k`` nearest historical matches to ``query``.

        Raises:
            RetrievalError: On dimension mismatch or search failure.
        """
        vector = self._validate_embeddings(query, expected_ndim=1).reshape(1, -1)
        vector = self._normalize_if_cosine(vector)
        try:
            distances, indices = self._index.search(vector, top_k)
        except Exception as exc:
            msg = f"FAISS search failed: {exc}"
            raise RetrievalError(msg) from exc
        matches: list[RetrievalMatch] = []
        for idx, dist in zip(indices[0], distances[0], strict=True):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[int(idx)]
            matches.append(
                RetrievalMatch(
                    vector_id=int(idx),
                    distance=float(dist),
                    dataset=str(meta.get("dataset", "")),
                    category=str(meta.get("category", "")),
                    defect_type=str(meta.get("defect_type", "")),
                    image_path=meta.get("image_path"),
                    metadata=dict(meta),
                )
            )
        return tuple(matches)

    def save(self, path: Path) -> None:
        """Persist the index and its metadata sidecar to ``path``.

        Raises:
            RetrievalError: On storage failure.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self._index, str(path))
            sidecar = {
                "index_metadata": self._metadata_meta.to_dict(),
                "records": self._metadata,
            }
            _sidecar_path(path).write_text(json.dumps(sidecar), encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to save FAISS index to {path}: {exc}"
            raise RetrievalError(msg) from exc

    def load(self, path: Path) -> None:
        """Load a previously saved index and its metadata sidecar from ``path``.

        Raises:
            RetrievalError: If the index is missing, corrupt, or was built
                with an incompatible embedding configuration.
        """
        path = Path(path)
        try:
            sidecar = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
            saved_meta = IndexMetadata.from_dict(sidecar["index_metadata"])
            if not self._metadata_meta.is_compatible_with(saved_meta):
                msg = (
                    f"Refusing to load index built with incompatible "
                    f"configuration: {saved_meta} != {self._metadata_meta}"
                )
                raise RetrievalError(msg)
            self._index = self._faiss.read_index(str(path))
            self._metadata = list(sidecar["records"])
        except RetrievalError:
            raise
        except Exception as exc:
            msg = f"Failed to load FAISS index from {path}: {exc}"
            raise RetrievalError(msg) from exc

    def _build_index(self) -> Any:
        """Construct the underlying FAISS index for the configured metric."""
        if self._metric == "l2":
            return self._faiss.IndexFlatL2(self._dim)
        return self._faiss.IndexFlatIP(self._dim)

    def _validate_embeddings(self, embeddings: Embedding, *, expected_ndim: int) -> Embedding:
        """Validate shape, dtype, and finiteness of ``embeddings``."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != expected_ndim:
            msg = f"Expected a {expected_ndim}D array, got shape {vectors.shape}"
            raise RetrievalError(msg)
        last_dim = vectors.shape[-1] if vectors.ndim > 0 else 0
        if vectors.size and last_dim != self._dim:
            msg = f"Expected embedding dimension {self._dim}, got {last_dim}"
            raise RetrievalError(msg)
        if vectors.size and not np.all(np.isfinite(vectors)):
            msg = "Embeddings contain NaN or infinite values"
            raise RetrievalError(msg)
        return vectors

    def _normalize_if_cosine(self, vectors: Embedding) -> Embedding:
        """L2-normalize rows when the index metric is ``"cosine"``."""
        if self._metric != "cosine" or vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return cast("Embedding", (vectors / norms).astype(np.float32))


def _sidecar_path(index_path: Path) -> Path:
    """Return the metadata sidecar path for a FAISS index file."""
    return index_path.with_name(index_path.name + _SIDECAR_SUFFIX)


def _import_faiss() -> Any:
    """Import FAISS lazily with a domain-specific error."""
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        msg = "faiss is not installed in the active environment"
        raise RetrievalError(msg) from exc


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    dim = 8
    index = FaissRetrievalIndex(dim, metric="cosine", embedding_model="smoke-test")

    base = rng.standard_normal((5, dim)).astype(np.float32)
    metadata = [
        {"dataset": "mvtec_ad", "category": "bottle", "defect_type": "crack"},
        {"dataset": "mvtec_ad", "category": "bottle", "defect_type": "scratch"},
        {"dataset": "visa", "category": "capsules", "defect_type": "poke"},
        {"dataset": "kolektorsdd2", "category": "surface", "defect_type": "blob"},
        {"dataset": "severstal", "category": "steel", "defect_type": "pit"},
    ]
    ids = index.add(base, metadata)
    assert ids == (0, 1, 2, 3, 4), ids

    query = base[2]
    results = index.search(query, top_k=1)
    assert results, "expected at least one match"
    assert results[0].vector_id == 2, f"nearest match should be itself, got {results[0]}"
    print(f"smoke test insert/query OK: nearest to row 2 is vector_id={results[0].vector_id}")

    tmp_path = Path("/tmp") / "faiss_index_smoke_test.faiss"
    index.save(tmp_path)

    reloaded = FaissRetrievalIndex(dim, metric="cosine", embedding_model="smoke-test")
    reloaded.load(tmp_path)
    reloaded_results = reloaded.search(query, top_k=1)
    assert reloaded_results[0].vector_id == 2, reloaded_results
    print("smoke test save/load OK")

    tmp_path.unlink(missing_ok=True)
    _sidecar_path(tmp_path).unlink(missing_ok=True)
