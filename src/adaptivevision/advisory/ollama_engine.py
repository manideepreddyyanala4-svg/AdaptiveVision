"""Local-LLM advisory engine backed by Ollama (Milestone M19).

Unlike :func:`adaptivevision.inference.onnx._import_onnxruntime`, a missing
``ollama`` package is not an error here - Ollama is an optional advisory
service, not a core production dependency, so a missing package or an
unreachable server both fall through to the same deterministic fallback path.
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any

from pydantic import ValidationError

from adaptivevision.advisory.schemas import RootCauseReportModel
from adaptivevision.common.interfaces import AdvisoryEngine
from adaptivevision.common.result import AdvisoryReport, InspectionEvidence

logger = logging.getLogger(__name__)

#: Default local model. Override via the ``model`` constructor argument.
DEFAULT_MODEL = "qwen2.5:7b"

_SYSTEM_PROMPT = (
    "You are a manufacturing quality-inspection assistant. You are given "
    "deterministic evidence about one inspected part. Respond ONLY with a "
    "JSON object matching the requested schema.\n"
    "Rules:\n"
    "- Do not invent evidence beyond what is supplied.\n"
    "- Clearly distinguish your hypothesis from established fact.\n"
    "- The severity has already been determined by a deterministic system "
    "and is not part of your response; do not restate or contradict it.\n"
    "- If you are uncertain, say so in root_cause_hypothesis and lower "
    "confidence_score accordingly."
)


class OllamaAdvisoryEngine(AdvisoryEngine):
    """An :class:`AdvisoryEngine` backed by a local Ollama LLM.

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
    return (
        f"Category: {evidence.category}\n"
        f"Anomaly score: {evidence.anomaly_score}\n"
        f"Deterministic severity: {evidence.severity.value}\n"
        f"Model version: {evidence.model_ver}\n"
        f"Historical similar defects (nearest first):\n{matches}\n\n"
        'Respond with JSON: {"defect_classification": str, '
        '"confidence_score": float in [0,1], "root_cause_hypothesis": str, '
        '"recommended_actions": [str, ...]}'
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
