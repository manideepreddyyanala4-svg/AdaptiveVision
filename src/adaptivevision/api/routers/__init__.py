"""HTTP route handlers grouped by resource (Milestone M15+).

This package is part of the frozen structure defined in Architecture
Specification v1.0. It is intentionally empty at Milestone M0; its modules
are implemented in the milestone noted above.
"""

from adaptivevision.api.routers.health import router as health_router
from adaptivevision.api.routers.metrics import router as metrics_router
from adaptivevision.api.routers.results import router as results_router

__all__ = ["health_router", "metrics_router", "results_router"]
