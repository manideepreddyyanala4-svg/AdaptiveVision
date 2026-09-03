"""HTTP route handlers for deployment-profile recommendations (Milestone M19).

The recommendation is always deterministic
(:func:`adaptivevision.deployment.profiles.recommend`) - this router never
involves an LLM, matching the architecture boundary that only the advisory
layer explains evidence and nothing here or upstream selects a model by
asking one.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from adaptivevision.deployment.profiles import (
    DeploymentProfile,
    explain_recommendation,
    feasible_profiles,
    recommend,
)

router = APIRouter(prefix="/api/v1/deployment", tags=["deployment"])


def get_deployment_profiles() -> tuple[DeploymentProfile, ...]:
    """Return the currently loaded, validated deployment profiles.

    The concrete profiles are injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@router.get("/profiles")
def list_deployment_profiles(
    profiles: Annotated[tuple[DeploymentProfile, ...], Depends(get_deployment_profiles)],
) -> dict[str, object]:
    """Return every loaded deployment profile."""
    return {"items": [p.to_dict() for p in profiles], "count": len(profiles)}


@router.get("/recommendation")
def get_recommendation(
    profiles: Annotated[tuple[DeploymentProfile, ...], Depends(get_deployment_profiles)],
    max_latency_ms: float = Query(..., gt=0),
    min_auroc: float = Query(..., ge=0.0, le=1.0),
    max_model_size_millions: float | None = Query(default=None, gt=0),
) -> dict[str, object]:
    """Return the deterministic recommended deployment configuration.

    Raises:
        HTTPException: 404 if no profile satisfies every constraint.
    """
    feasible = feasible_profiles(
        profiles,
        max_latency_ms=max_latency_ms,
        min_auroc=min_auroc,
        max_model_size_millions=max_model_size_millions,
    )
    picked = recommend(
        profiles,
        max_latency_ms=max_latency_ms,
        min_auroc=min_auroc,
        max_model_size_millions=max_model_size_millions,
    )
    if picked is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No deployment profile satisfies the given constraints",
        )
    return {
        "profile": picked.to_dict(),
        "reason": explain_recommendation(
            picked,
            max_latency_ms=max_latency_ms,
            min_auroc=min_auroc,
            n_feasible=len(feasible),
        ),
    }
