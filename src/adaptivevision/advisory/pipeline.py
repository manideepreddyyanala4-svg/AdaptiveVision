"""Advisory pipeline orchestration (Milestone M19).

Wires the deterministic decision output and the advisory engine together in
the fixed order the architecture requires: :func:`build_evidence` only reads
an already-final :class:`~adaptivevision.decision.policy.Decision`, never
recomputes or influences it, and :func:`advise` enforces that whatever an
:class:`~adaptivevision.common.interfaces.AdvisoryEngine` returns still
carries the same severity it was given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adaptivevision.common.enums import Severity
from adaptivevision.common.errors import AdvisoryError
from adaptivevision.common.result import InspectionEvidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from adaptivevision.common.interfaces import AdvisoryEngine
    from adaptivevision.common.result import AdvisoryReport, Defect, RetrievalMatch
    from adaptivevision.decision.policy import Decision

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
) -> InspectionEvidence:
    """Build read-only advisory evidence from an already-final ``decision``.

    Args:
        inspection_id: Identifier of the inspected sample.
        category: Product/part category being inspected.
        anomaly_score: Overall anomaly score, if computed.
        model_ver: Version of the anomaly model that produced the score.
        decision: The final, already-computed decision. Never recomputed.
        retrieval_matches: Historical matches retrieved for this sample.

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
            practice given :class:`~adaptivevision.advisory.ollama_engine.OllamaAdvisoryEngine`
            always echoes it.
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
