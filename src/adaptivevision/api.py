"""Serve stage: the HTTP API and dashboard page.

The FastAPI application factory plus every route: a health check, read-only
inspection results, runtime metrics (JSON and Prometheus text exposition),
component health, advisory reports (Milestone M19), deployment-profile
recommendations, a live WebSocket results feed, and a minimal dashboard page.

Every dependency (the result repository, metrics registry, health check,
advisory repository, deployment profiles) is injected at the composition
root (``app.py``) so this module can be exercised against any backend.
:class:`~adaptivevision.common.AdvisoryRepository` and the loaded deployment
profiles are both optional: the API runs without them, exposing only the
routes that need them.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import HTMLResponse, PlainTextResponse

from adaptivevision.common import AdvisoryRepository, InspectionResult, ResultRepository
from adaptivevision.deployment import (
    DeploymentProfile,
    explain_recommendation,
    feasible_profiles,
    recommend,
)
from adaptivevision.drift import HealthCheck, MetricsRegistry, render_metrics

# =============================================================================
# Routers
# =============================================================================

results_router = APIRouter(prefix="/api/v1/results", tags=["results"])


def get_repository() -> ResultRepository:
    """Return the result repository backing the results endpoints.

    The concrete repository is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


def _to_dict(result: InspectionResult) -> dict[str, object]:
    """Convert an inspection result to a JSON-serializable mapping."""
    return result.to_dict()


@results_router.get("")
def list_results(
    repository: Annotated[ResultRepository, Depends(get_repository)],
    limit: int = 100,
    offset: int = 0,
) -> dict[str, object]:
    """Return a page of inspection results, most-recent first."""
    results = repository.list_results(limit=limit, offset=offset)
    return {"items": [_to_dict(r) for r in results], "count": len(results)}


@results_router.get("/{inspection_id}")
def get_result(
    inspection_id: str,
    repository: Annotated[ResultRepository, Depends(get_repository)],
) -> dict[str, object]:
    """Return a single inspection result by identifier."""
    result = repository.get_result(inspection_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {inspection_id} not found",
        )
    return _to_dict(result)


metrics_router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


def get_metrics() -> MetricsRegistry:
    """Return the metrics registry backing the metrics endpoints.

    The concrete registry is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@metrics_router.get("")
def metrics_snapshot(
    registry: Annotated[MetricsRegistry, Depends(get_metrics)],
) -> dict[str, object]:
    """Return a snapshot of all runtime metrics."""
    return registry.snapshot()


health_router = APIRouter(prefix="/api/v1/health", tags=["health"])


def get_health() -> HealthCheck:
    """Return the health check backing the health endpoints.

    The concrete health check is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@health_router.get("")
def health_status(
    health: Annotated[HealthCheck, Depends(get_health)],
) -> dict[str, object]:
    """Return the health status of all registered components."""
    return {
        "healthy": health.is_healthy(),
        "components": [
            {"name": status_.name, "healthy": status_.healthy, "detail": status_.detail}
            for status_ in health.check()
        ],
    }


advisory_router = APIRouter(prefix="/api/v1/advisory", tags=["advisory"])


def get_advisory_repository() -> AdvisoryRepository:
    """Return the advisory repository backing the advisory endpoints.

    The concrete repository is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@advisory_router.get("/{inspection_id}")
def get_advisory_report(
    inspection_id: str,
    repository: Annotated[AdvisoryRepository, Depends(get_advisory_repository)],
) -> dict[str, object]:
    """Return the advisory report for ``inspection_id``, if one was produced."""
    report = repository.get_report(inspection_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No advisory report for inspection {inspection_id}",
        )
    return report.to_dict()


deployment_router = APIRouter(prefix="/api/v1/deployment", tags=["deployment"])


def get_deployment_profiles() -> tuple[DeploymentProfile, ...]:
    """Return the currently loaded, validated deployment profiles.

    The concrete profiles are injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@deployment_router.get("/profiles")
def list_deployment_profiles(
    profiles: Annotated[tuple[DeploymentProfile, ...], Depends(get_deployment_profiles)],
) -> dict[str, object]:
    """Return every loaded deployment profile."""
    return {"items": [p.to_dict() for p in profiles], "count": len(profiles)}


@deployment_router.get("/recommendation")
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


# =============================================================================
# Application factory
# =============================================================================

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AdaptiveVision Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
    th { background: #f0f0f0; }
  </style>
</head>
<body>
  <h1>AdaptiveVision Dashboard</h1>
  <p>Read-only view of persisted inspection results.</p>
  <table id="results">
    <thead>
      <tr><th>Inspection</th><th>Part</th><th>Verdict</th><th>Cycle (ms)</th><th>Advisory</th></tr>
    </thead>
    <tbody></tbody>
  </table>
  <script>
    // Advisory reports (Milestone M19) are fetched per-row, best-effort: a
    // 404 (no report yet, or the advisory route isn't registered at all
    // when the station has no advisory repository configured) just leaves
    // the cell at its placeholder rather than failing the row.
    function loadAdvisory(inspectionId, cell) {
      fetch('/api/v1/advisory/' + encodeURIComponent(inspectionId))
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (report) {
          if (report) {
            cell.textContent = report.severity + ': ' + report.defect_classification;
          }
        })
        .catch(function () { /* advisory unavailable; leave placeholder */ });
    }

    fetch('/api/v1/results?limit=50')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var body = document.querySelector('#results tbody');
        data.items.forEach(function (item) {
          var row = document.createElement('tr');
          row.innerHTML = '<td>' + item.inspection_id + '</td>' +
            '<td>' + item.part_id + '</td>' +
            '<td>' + item.verdict + '</td>' +
            '<td>' + item.cycle_time_ms.toFixed(1) + '</td>' +
            '<td>-</td>';
          body.appendChild(row);
          loadAdvisory(item.inspection_id, row.lastElementChild);
        });
      });
  </script>
</body>
</html>
"""


class _LiveHub:
    """Broadcasts inspection results to connected WebSocket clients."""

    def __init__(self) -> None:
        """Initialize an empty hub."""
        self._clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a WebSocket client."""
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket client."""
        self._clients.discard(websocket)

    async def broadcast(self, payload: dict[str, object]) -> None:
        """Send ``payload`` to every connected client."""
        for client in list(self._clients):
            await client.send_json(payload)


def create_app(
    repository: ResultRepository,
    *,
    metrics: MetricsRegistry | None = None,
    health: HealthCheck | None = None,
    advisory: AdvisoryRepository | None = None,
    deployment_profiles: tuple[DeploymentProfile, ...] | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        repository: The result repository backing the results endpoints.
        metrics: The metrics registry backing the metrics endpoints.
        health: The health check backing the health endpoints.
        advisory: The advisory repository backing the advisory endpoints
            (Milestone M19). When ``None`` (the default), the advisory routes
            are not registered at all - a fully supported configuration.
        deployment_profiles: The loaded, validated deployment profiles
            backing the deployment endpoints (Milestone M19). Defaults to no
            profiles, in which case ``/api/v1/deployment/recommendation``
            always reports no feasible candidate.

    Returns:
        The configured FastAPI application.
    """
    metrics = metrics or MetricsRegistry()
    health = health or HealthCheck()
    deployment_profiles = deployment_profiles or ()
    hub = _LiveHub()

    app = FastAPI(title="AdaptiveVision", version="0.0.0")

    def _get_repository() -> ResultRepository:
        return repository

    def _get_metrics() -> MetricsRegistry:
        return metrics

    def _get_health() -> HealthCheck:
        return health

    def _get_deployment_profiles() -> tuple[DeploymentProfile, ...]:
        return deployment_profiles

    app.dependency_overrides[get_repository] = _get_repository
    app.dependency_overrides[get_metrics] = _get_metrics
    app.dependency_overrides[get_health] = _get_health
    app.dependency_overrides[get_deployment_profiles] = _get_deployment_profiles
    app.include_router(results_router)
    app.include_router(metrics_router)
    app.include_router(health_router)
    app.include_router(deployment_router)

    if advisory is not None:

        def _get_advisory_repository() -> AdvisoryRepository:
            return advisory

        app.dependency_overrides[get_advisory_repository] = _get_advisory_repository
        app.include_router(advisory_router)

    @app.get("/health")
    def health_endpoint() -> dict[str, str]:
        """Return the service health status."""
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        """Serve runtime metrics in Prometheus text exposition format.

        This endpoint is scraped by Prometheus for edge observability.
        """
        return render_metrics(metrics)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        """Serve the dashboard page."""
        return _DASHBOARD_HTML

    @app.websocket("/ws/results")
    async def results_ws(websocket: WebSocket) -> None:
        """Stream live inspection results to connected clients."""
        await hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    return app
