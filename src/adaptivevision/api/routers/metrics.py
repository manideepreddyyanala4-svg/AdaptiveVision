"""HTTP route handlers for runtime metrics (Milestone M15)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from adaptivevision.monitoring import MetricsRegistry

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


def get_metrics() -> MetricsRegistry:
    """Return the metrics registry backing the metrics endpoints.

    The concrete registry is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@router.get("")
def metrics_snapshot(
    registry: Annotated[MetricsRegistry, Depends(get_metrics)],
) -> dict[str, object]:
    """Return a snapshot of all runtime metrics."""
    return registry.snapshot()
