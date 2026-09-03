"""FastAPI application factory (Milestone M15; M19 adds advisory/deployment).

The API exposes a health check, read-only inspection results, runtime metrics,
component health, and a dashboard page. The :class:`ResultRepository`,
:class:`MetricsRegistry`, and :class:`HealthCheck` are injected at the
composition root so the API can be exercised against any backend.
:class:`~adaptivevision.common.interfaces.AdvisoryRepository` and the loaded
:class:`~adaptivevision.deployment.profiles.DeploymentProfile` tuple are both
optional (Milestone M19): the API runs without them, exposing only the
routes that need them.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

from adaptivevision.api.routers.advisory import get_advisory_repository
from adaptivevision.api.routers.advisory import router as advisory_router
from adaptivevision.api.routers.deployment import get_deployment_profiles
from adaptivevision.api.routers.deployment import router as deployment_router
from adaptivevision.api.routers.health import get_health
from adaptivevision.api.routers.health import router as health_router
from adaptivevision.api.routers.metrics import get_metrics
from adaptivevision.api.routers.metrics import router as metrics_router
from adaptivevision.api.routers.results import get_repository
from adaptivevision.api.routers.results import router as results_router
from adaptivevision.common.interfaces import AdvisoryRepository, ResultRepository
from adaptivevision.deployment.profiles import DeploymentProfile
from adaptivevision.monitoring import HealthCheck, MetricsRegistry, render_metrics

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
  <p>Full dashboard (Milestone M15).</p>
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

        This endpoint is scraped by Prometheus for edge observability (M18).
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
