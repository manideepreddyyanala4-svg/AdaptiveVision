"""Integration tests for the M13 minimal HTTP API."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from adaptivevision.api import create_app
from adaptivevision.common import Severity, Verdict
from adaptivevision.common import AdvisoryRepository, ResultRepository
from adaptivevision.common import AdvisoryReport, InspectionEvidence, InspectionResult
from adaptivevision.deployment import DeploymentProfile
from adaptivevision.drift import HealthCheck, MetricsRegistry


class _FakeAdvisoryRepository(AdvisoryRepository):
    def __init__(self, reports: dict[str, AdvisoryReport] | None = None) -> None:
        self._reports = dict(reports or {})

    def save_report(
        self, inspection_id: str, evidence: InspectionEvidence, report: AdvisoryReport
    ) -> None:
        self._reports[inspection_id] = report

    def get_report(self, inspection_id: str) -> AdvisoryReport | None:
        return self._reports.get(inspection_id)


class _FakeRepository(ResultRepository):
    def __init__(self, results: list[InspectionResult]) -> None:
        self._results = {r.inspection_id: r for r in results}

    def save_result(self, result: InspectionResult) -> None:
        self._results[result.inspection_id] = result

    def get_result(self, inspection_id: str) -> InspectionResult | None:
        return self._results.get(inspection_id)

    def list_results(self, *, limit: int = 100, offset: int = 0) -> tuple[InspectionResult, ...]:
        ordered = sorted(self._results.values(), key=lambda r: r.timestamp_utc, reverse=True)
        return tuple(ordered[offset : offset + limit])


def _make_result(inspection_id: str, part_id: str, verdict: Verdict) -> InspectionResult:
    return InspectionResult(
        inspection_id=inspection_id,
        part_id=part_id,
        station_id="station-1",
        verdict=verdict,
        recipe_ver="1.0.0",
        model_ver="1.0.0",
        calib_ver="1.0.0",
        cycle_time_ms=12.5,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _client() -> TestClient:
    repository = _FakeRepository(
        [
            _make_result("insp-1", "part-1", Verdict.PASS),
            _make_result("insp-2", "part-2", Verdict.FAIL),
        ]
    )
    return TestClient(create_app(repository))


def test_health_endpoint() -> None:
    with _client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_list_results() -> None:
    with _client() as client:
        response = client.get("/api/v1/results")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert {item["inspection_id"] for item in body["items"]} == {"insp-1", "insp-2"}


def test_get_result() -> None:
    with _client() as client:
        response = client.get("/api/v1/results/insp-1")
        assert response.status_code == 200
        body = response.json()
        assert body["inspection_id"] == "insp-1"
        assert body["part_id"] == "part-1"
        assert body["verdict"] == Verdict.PASS.value


def test_get_missing_result_returns_404() -> None:
    with _client() as client:
        response = client.get("/api/v1/results/nope")
        assert response.status_code == 404


def test_dashboard_served_at_root() -> None:
    with _client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "AdaptiveVision Dashboard" in response.text


def test_metrics_endpoint() -> None:
    metrics = MetricsRegistry()
    metrics.increment("inspections")
    repository = _FakeRepository([])
    with TestClient(create_app(repository, metrics=metrics)) as client:
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        assert response.json()["counters"] == {"inspections": 1}


def test_prometheus_metrics_endpoint() -> None:
    metrics = MetricsRegistry()
    metrics.increment("inspections", 2)
    metrics.set_gauge("temperature", 42.0)
    repository = _FakeRepository([])
    with TestClient(create_app(repository, metrics=metrics)) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        body = response.text
        assert "inspections_total 2" in body
        assert "temperature 42" in body


def test_health_status_endpoint() -> None:
    health = HealthCheck()
    health.register("camera", lambda: True)

    repository = _FakeRepository([])
    with TestClient(create_app(repository, health=health)) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["healthy"] is True
        assert body["components"][0]["name"] == "camera"


# ---------------------------------------------------------------------------
# Advisory and deployment routes (Milestone M19)
# ---------------------------------------------------------------------------


def _profile(model: str, *, auroc: float, p95: float) -> DeploymentProfile:
    return DeploymentProfile(
        model=model,
        family="memory_bank",
        backbone="resnet50",
        config=f"{model}-cfg",
        dataset="mvtec_ad",
        n_seeds=3,
        benchmark_version="2026-08-26",
        validated_at="2026-08-26T00:00:00+00:00",
        image_auroc=auroc,
        p95_latency_ms=p95,
    )


def test_advisory_route_not_registered_when_repository_absent() -> None:
    repository = _FakeRepository([])
    with TestClient(create_app(repository)) as client:
        response = client.get("/api/v1/advisory/insp-1")
        assert response.status_code == 404


def test_advisory_route_returns_report_when_present() -> None:
    report = AdvisoryReport(
        defect_classification="crack",
        severity=Severity.MAJOR,
        confidence_score=0.8,
        root_cause_hypothesis="Likely mold defect.",
        recommended_actions=("inspect mold",),
    )
    advisory = _FakeAdvisoryRepository({"insp-1": report})
    repository = _FakeRepository([])
    with TestClient(create_app(repository, advisory=advisory)) as client:
        response = client.get("/api/v1/advisory/insp-1")
        assert response.status_code == 200
        assert response.json()["defect_classification"] == "crack"


def test_advisory_route_missing_report_returns_404() -> None:
    advisory = _FakeAdvisoryRepository()
    repository = _FakeRepository([])
    with TestClient(create_app(repository, advisory=advisory)) as client:
        response = client.get("/api/v1/advisory/insp-1")
        assert response.status_code == 404


def test_deployment_profiles_route_lists_loaded_profiles() -> None:
    repository = _FakeRepository([])
    profiles = (_profile("patchcore", auroc=0.99, p95=17.3),)
    with TestClient(create_app(repository, deployment_profiles=profiles)) as client:
        response = client.get("/api/v1/deployment/profiles")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        assert body["items"][0]["model"] == "patchcore"


def test_deployment_recommendation_route_returns_reasoned_pick() -> None:
    repository = _FakeRepository([])
    profiles = (
        _profile("slow", auroc=0.995, p95=90.0),
        _profile("fast", auroc=0.95, p95=15.0),
    )
    with TestClient(create_app(repository, deployment_profiles=profiles)) as client:
        response = client.get(
            "/api/v1/deployment/recommendation",
            params={"max_latency_ms": 50.0, "min_auroc": 0.9},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["profile"]["model"] == "fast"
        assert "fast" in body["reason"]


def test_deployment_recommendation_route_404_when_infeasible() -> None:
    repository = _FakeRepository([])
    profiles = (_profile("only", auroc=0.5, p95=200.0),)
    with TestClient(create_app(repository, deployment_profiles=profiles)) as client:
        response = client.get(
            "/api/v1/deployment/recommendation",
            params={"max_latency_ms": 50.0, "min_auroc": 0.9},
        )
        assert response.status_code == 404


def test_deployment_profiles_route_empty_by_default() -> None:
    repository = _FakeRepository([])
    with TestClient(create_app(repository)) as client:
        response = client.get("/api/v1/deployment/profiles")
        assert response.status_code == 200
        assert response.json() == {"items": [], "count": 0}


# ---------------------------------------------------------------------------
# Live-results WebSocket hub (Milestone M15, pre-existing but untested)
# ---------------------------------------------------------------------------


def test_results_websocket_connects_and_disconnects_cleanly() -> None:
    repository = _FakeRepository([])
    with (
        TestClient(create_app(repository)) as client,
        client.websocket_connect("/ws/results") as websocket,
    ):
        websocket.close()


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)


def test_live_hub_broadcasts_to_connected_clients() -> None:
    import asyncio

    from adaptivevision.api import _LiveHub

    hub = _LiveHub()
    client_a, client_b = _FakeWebSocket(), _FakeWebSocket()
    hub._clients.add(client_a)  # type: ignore[arg-type]
    hub._clients.add(client_b)  # type: ignore[arg-type]

    asyncio.run(hub.broadcast({"inspection_id": "insp-1"}))

    assert client_a.sent == [{"inspection_id": "insp-1"}]
    assert client_b.sent == [{"inspection_id": "insp-1"}]

    hub.disconnect(client_a)  # type: ignore[arg-type]
    asyncio.run(hub.broadcast({"inspection_id": "insp-2"}))
    assert client_a.sent == [{"inspection_id": "insp-1"}]
    assert client_b.sent == [{"inspection_id": "insp-1"}, {"inspection_id": "insp-2"}]
