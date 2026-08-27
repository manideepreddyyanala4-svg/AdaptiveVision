"""HTTP route handlers for component health (Milestone M15)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from adaptivevision.monitoring import HealthCheck

router = APIRouter(prefix="/api/v1/health", tags=["health"])


def get_health() -> HealthCheck:
    """Return the health check backing the health endpoints.

    The concrete health check is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@router.get("")
def health_status(
    health: Annotated[HealthCheck, Depends(get_health)],
) -> dict[str, object]:
    """Return the health status of all registered components."""
    return {
        "healthy": health.is_healthy(),
        "components": [
            {"name": status.name, "healthy": status.healthy, "detail": status.detail}
            for status in health.check()
        ],
    }
