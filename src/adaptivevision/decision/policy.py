"""Verdict fusion and decision policy (Milestone M10).

The decision policy fuses the partial results produced by the inspection
inspectors (metrology M7, anomaly M9, classical AOI) into a single verdict.
It follows the deterministic inspection path: defects are never silently
dropped, and a part is only PASS when no inspector raised a defect.

Severity rules:
- Any CRITICAL or MAJOR defect -> FAIL.
- Otherwise any MINOR or INFO defect -> REVIEW.
- Otherwise, if an anomaly score falls in a configured review band -> REVIEW.
- Otherwise -> PASS.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from adaptivevision.common.enums import Severity, Verdict
from adaptivevision.common.result import AnomalyResult, Defect, PartialResult


@dataclass(frozen=True, slots=True)
class Decision:
    """The fused outcome of an inspection.

    Attributes:
        verdict: The final verdict.
        defects: All defects fused from the partial results.
    """

    verdict: Verdict
    defects: tuple[Defect, ...]


class DecisionPolicy:
    """Fuse partial inspection results into a single verdict.

    Args:
        anomaly_review_threshold: Optional anomaly score at or above which a
            non-anomalous part is routed to REVIEW instead of PASS.
    """

    def __init__(self, *, anomaly_review_threshold: float | None = None) -> None:
        """Initialize the policy."""
        self._anomaly_review_threshold = anomaly_review_threshold

    def decide(self, partials: Iterable[PartialResult]) -> Decision:
        """Fuse partial results into a verdict.

        Args:
            partials: The partial results from the inspection inspectors.

        Returns:
            A :class:`Decision` with the fused verdict and defects.
        """
        defects: list[Defect] = []
        for partial in partials:
            defects.extend(getattr(partial, "defects", ()))

        verdict = self._verdict_for(defects, partials)

        return Decision(verdict=verdict, defects=tuple(defects))

    def _verdict_for(
        self,
        defects: list[Defect],
        partials: Iterable[PartialResult],
    ) -> Verdict:
        """Determine the verdict from fused defects and partial results."""
        for defect in defects:
            if defect.severity in (Severity.CRITICAL, Severity.MAJOR):
                return Verdict.FAIL
        if any(
            defect.severity in (Severity.MINOR, Severity.INFO) for defect in defects
        ):
            return Verdict.REVIEW

        if self._anomaly_review_threshold is not None:
            for partial in partials:
                if (
                    isinstance(partial, AnomalyResult)
                    and not partial.is_anomalous
                    and partial.score >= self._anomaly_review_threshold
                ):
                    return Verdict.REVIEW

        return Verdict.PASS
