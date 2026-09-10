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
import queue
import threading
import time
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
    AdvisoryRepository,
    DefectMeasurement,
    InspectionEvidence,
    InspectionResult,
    RetrievalError,
    RetrievalIndex,
    RetrievalMatch,
    Severity,
    Verdict,
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
    measurements: tuple[DefectMeasurement, ...] = (),
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
        measurements: Physical defect metrology (microns squared, aspect
            ratio, morphology) measured for this inspection, if any.

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
        measurements=measurements,
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


# =============================================================================
# Asynchronous advisory worker
#
# The advisory layer is explanatory, not decisional, so it must never sit on
# the inspection critical path. This worker takes finished results on a
# bounded queue and does the LLM call on its own thread: submit() only
# enqueues, so the station's cycle time is unaffected whether the model
# answers in 200 ms or 20 s (or not at all).
# =============================================================================

#: Verdicts worth explaining. A clean PASS needs no root-cause analysis.
ADVISORY_VERDICTS: tuple[Verdict, ...] = (Verdict.FAIL, Verdict.REVIEW)

#: Seconds a graceful stop waits for in-flight advisory work. Sized for a
#: local model's response time, not a network call's.
DEFAULT_STOP_TIMEOUT_S = 30.0

#: Bounded queue depth. When full, the oldest pending job is dropped rather
#: than blocking the caller -- the same never-block-the-line rule the camera
#: frame buffer follows.
DEFAULT_QUEUE_SIZE = 64


class AdvisoryWorker:
    """Runs advisory generation off the inspection thread.

    Reports are written from the worker thread, so ``repository`` must be
    backed by a store that tolerates cross-thread use: a file-backed SQLite
    database does, an in-memory one does not (SQLAlchemy gives each thread
    its own ``:memory:`` database).

    Args:
        engine: The advisory engine to generate reports with.
        repository: Where finished reports are persisted.
        category: Product category recorded in the evidence.
        queue_size: Maximum number of pending jobs.
        verdicts: Verdicts that trigger an advisory.
        stop_timeout: Seconds :meth:`stop` waits for in-flight work before
            giving up. A local model can take tens of seconds, so a
            short-lived process needs a budget well above the call latency
            or it discards work it already started.
    """

    def __init__(
        self,
        engine: AdvisoryEngine,
        repository: AdvisoryRepository,
        *,
        category: str = "",
        queue_size: int = DEFAULT_QUEUE_SIZE,
        verdicts: tuple[Verdict, ...] = ADVISORY_VERDICTS,
        stop_timeout: float = DEFAULT_STOP_TIMEOUT_S,
    ) -> None:
        """Initialize the worker without starting its thread."""
        self._engine = engine
        self._repository = repository
        self._category = category
        self._verdicts = verdicts
        self._stop_timeout = stop_timeout
        self._queue: queue.Queue[InspectionResult | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        # Tracks jobs accepted but not yet finished, so drain() can wait for
        # completion rather than for the queue merely to empty -- a job is
        # off the queue for the whole duration of its LLM call.
        self._idle = threading.Condition()
        self._pending = 0

    def start(self) -> None:
        """Start the background thread. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="advisory-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Finish pending work, then stop the thread.

        Waits for in-flight jobs before signalling exit, so a short-lived
        process does not throw away an advisory it has already started
        generating.

        Args:
            timeout: Seconds to wait. Defaults to the worker's configured
                ``stop_timeout``.
        """
        if self._thread is None:
            return
        budget = self._stop_timeout if timeout is None else timeout
        if not self.drain(budget):
            logger.warning("Advisory worker still busy after %.1fs; abandoning", budget)
        self._queue.put(None)
        self._thread.join(budget)
        self._thread = None

    def submit(self, result: InspectionResult) -> bool:
        """Enqueue ``result`` for advisory generation.

        Never blocks and never raises: this is called from the inspection
        loop, which must not be delayed by advisory work.

        Args:
            result: The completed inspection result.

        Returns:
            ``True`` if the job was enqueued, ``False`` if it was skipped
            (verdict not of interest) or dropped (queue full).
        """
        if result.verdict not in self._verdicts:
            return False
        try:
            with self._idle:
                self._queue.put_nowait(result)
                self._pending += 1
        except queue.Full:
            logger.warning(
                "Advisory queue full; dropping job",
                extra={"inspection_id": result.inspection_id},
            )
            return False
        return True

    def on_result(self, result: InspectionResult) -> None:
        """``on_result``-compatible adapter over :meth:`submit`."""
        self.submit(result)

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until every accepted job has finished (tests/shutdown).

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            ``True`` if the worker went idle, ``False`` on timeout.
        """
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        return True

    def _run(self) -> None:
        """Consume jobs until stopped."""
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._process(job)
            except Exception as exc:
                logger.error(
                    "Advisory generation failed",
                    extra={"inspection_id": job.inspection_id, "error": str(exc)},
                )
            finally:
                self._queue.task_done()
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def _process(self, result: InspectionResult) -> None:
        """Generate and persist one advisory report."""
        evidence = InspectionEvidence(
            sample_id=result.inspection_id,
            category=self._category,
            anomaly_score=result.anomaly_score,
            severity=_max_severity(result.defects),
            model_ver=result.model_ver,
            measurements=result.defect_measurements,
            heatmap_region=result.heatmap_region,
        )
        report = advise(evidence, advisory=self._engine)
        if report is None:
            return
        self._repository.save_report(result.inspection_id, evidence, report)
        logger.info(
            "Advisory report stored",
            extra={
                "inspection_id": result.inspection_id,
                "is_fallback": report.is_fallback,
            },
        )


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

#: Default Ollama endpoint. Loopback by design: the advisory model runs on
#: the edge IPC itself, so no inspection evidence ever leaves the machine.
DEFAULT_HOST = "http://127.0.0.1:11434"

#: Default per-attempt wall-clock budget, in seconds. A local model that has
#: not answered within this budget is abandoned in favour of the
#: deterministic fallback: the advisory layer is explanatory only, and must
#: never hold up the station's disposition of a part.
DEFAULT_TIMEOUT_S = 0.5

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
    """An :class:`~adaptivevision.common.AdvisoryEngine` backed by a local Ollama LLM.

    Never raises: a missing ``ollama`` package, an unreachable server, a
    timeout, or a malformed response all fall back to a deterministic report
    derived only from the supplied evidence.

    Args:
        model: Ollama model identifier.
        host: Base URL of the local Ollama server.
        timeout_s: Per-attempt wall-clock budget. Exceeding it is treated as
            a failed attempt, not an error.
        max_retries: Number of retries after a malformed structured response,
            before falling back.
        client: Optional Ollama-client-like object, used by tests.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        """Initialize the engine without connecting to a server."""
        self._model = model
        self._host = host
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client = client

    def generate_report(self, evidence: InspectionEvidence) -> AdvisoryReport:
        """Produce a validated advisory report, falling back if the LLM is unavailable."""
        client = self._client if self._client is not None else self._build_client()
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

    def _build_client(self) -> Any | None:
        """Build a host- and timeout-bound Ollama client, or ``None``.

        The timeout is bound to the client itself so a hung or overloaded
        local server can never stall the caller past ``timeout_s``.
        """
        module = _try_import_ollama()
        if module is None:
            return None
        try:
            return module.Client(host=self._host, timeout=self._timeout_s)
        except Exception as exc:  # pragma: no cover - defensive, client ctor
            logger.warning("Could not build Ollama client for %s: %s", self._host, exc)
            return None


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
    metrology = (
        "\n".join(
            f"  - {m.morphology}: area={m.area_um2:.1f} um^2, "
            f"aspect_ratio={m.aspect_ratio:.2f}, bbox={m.bbox}"
            for m in evidence.measurements
        )
        or "  (none measured)"
    )
    return (
        f"Category: {evidence.category}\n"
        f"Anomaly score: {evidence.anomaly_score}\n"
        f"Deterministic severity: {evidence.severity.value}\n"
        f"Model version: {evidence.model_ver}\n"
        f"{region_line}"
        f"Measured defect geometry (calibrated microns, largest first):\n"
        f"{metrology}\n"
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
    largest = evidence.measurements[0] if evidence.measurements else None
    hypothesis = f"No LLM analysis available. Deterministic severity is {evidence.severity.value}."
    if largest is not None:
        hypothesis += (
            f" Largest measured region: {largest.morphology}, "
            f"{largest.area_um2:.1f} um^2, aspect ratio {largest.aspect_ratio:.2f}."
        )
    if evidence.heatmap_region:
        hypothesis += f" Anomaly signal concentrated in the {evidence.heatmap_region}."
    if nearest is not None:
        hypothesis += (
            f" Most similar historical defect on record: {nearest.defect_type} "
            f"in {nearest.dataset}/{nearest.category}."
        )
    classification = "unknown"
    if largest is not None:
        classification = largest.morphology
    elif nearest is not None:
        classification = nearest.defect_type
    return AdvisoryReport(
        defect_classification=classification,
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
