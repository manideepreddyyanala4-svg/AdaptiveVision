"""Pydantic validation for LLM-produced advisory reports (Milestone M19).

This is the only place Pydantic is used for validation. A raw LLM response is
untrusted input and must be validated before use; once validated it is
converted to the stable :class:`~adaptivevision.common.result.AdvisoryReport`
frozen dataclass, so the rest of the system (persistence, API) continues to
work with the same frozen-dataclass value objects everywhere else (frozen
decisions 4/5). Deliberately absent from this schema: a ``severity`` field -
severity is never something the LLM is asked or allowed to set, it is always
echoed unchanged from the deterministic evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
