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
from typing import TYPE_CHECKING

from adaptivevision.common.enums import Severity, Verdict
from adaptivevision.common.result import AnomalyResult, Defect, PartialResult

if TYPE_CHECKING:
    from adaptivevision.recipe.model import Recipe

#: Ordering used to compare severities against a configurable
#: ``fail_severity`` threshold. ``Severity`` is a plain ``StrEnum`` (its
#: string values are stable API, e.g. persisted in recipes/results), so this
#: rank map -- not enum declaration order -- is what "at or above" means.
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.MINOR: 1,
    Severity.MAJOR: 2,
    Severity.CRITICAL: 3,
}


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
        fail_severity: Minimum defect severity that forces FAIL rather than
            REVIEW. Defaults to MAJOR (CRITICAL always fails regardless).
        max_defects: Optional cap on tolerated defect count; a part with more
            defects than this fails even if every individual defect is below
            ``fail_severity``.
    """

    def __init__(
        self,
        *,
        anomaly_review_threshold: float | None = None,
        fail_severity: Severity = Severity.MAJOR,
        max_defects: int | None = None,
    ) -> None:
        """Initialize the policy."""
        self._anomaly_review_threshold = anomaly_review_threshold
        self._fail_severity = fail_severity
        self._max_defects = max_defects

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> DecisionPolicy:
        """Build a live policy from a recipe's *declared* decision contract.

        ``recipe.decision`` (:class:`adaptivevision.recipe.model.DecisionPolicy`,
        Milestone M2) is a data-only description of the desired rules; this
        engine (Milestone M10) is what actually applies them. ``max_defects``
        and ``fail_severity`` map directly. ``review_on_anomaly`` doesn't map
        onto this policy at all -- it's honored by the anomaly *detector*
        instead (see ``ThresholdAnomalyDetector(anomalous_severity=...)``),
        since it decides what severity an anomaly defect carries in the first
        place, not how an already-built defect list is judged.

        Args:
            recipe: The recipe whose declared policy to translate.

        Returns:
            A configured :class:`DecisionPolicy`.
        """
        declared = recipe.decision
        return cls(
            fail_severity=declared.fail_severity,
            max_defects=declared.max_defects or None,
        )

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
        fail_rank = _SEVERITY_RANK[self._fail_severity]
        for defect in defects:
            if _SEVERITY_RANK[defect.severity] >= fail_rank:
                return Verdict.FAIL
        if self._max_defects is not None and len(defects) > self._max_defects:
            return Verdict.FAIL
        if defects:
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
