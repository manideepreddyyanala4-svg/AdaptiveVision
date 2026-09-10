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

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from adaptivevision.common import (
    AdaptiveVisionError,
    AdvisoryRepository,
    InspectionResult,
    ResultRepository,
)
from adaptivevision.deployment import (
    DeploymentProfile,
    explain_recommendation,
    feasible_profiles,
    recommend,
)
from adaptivevision.drift import HealthCheck, MetricsRegistry, render_metrics
from adaptivevision.orchestration import ManualInspectionService
from adaptivevision.storage import ImageStoreError, LocalImageStore

if TYPE_CHECKING:
    from adaptivevision.common import Image

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
# Archived inspection images
#
# Serves the frames the station archived for each part so an operator can
# see what was actually inspected, not just the numbers derived from it.
# =============================================================================

images_router = APIRouter(prefix="/api/v1/images", tags=["images"])


def get_image_store() -> LocalImageStore:
    """Return the image store backing the image endpoints.

    The concrete store is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


@images_router.get("/{image_id}")
def get_image(
    image_id: str,
    store: Annotated[LocalImageStore, Depends(get_image_store)],
) -> FileResponse:
    """Serve one archived image as PNG."""
    try:
        path = store.path_for(image_id)
    except ImageStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return FileResponse(path, media_type="image/png")


# =============================================================================
# Manual (engineering / teach) ingestion
#
# Lets an engineer put one sample through the full inspection path from the
# console, for setup and diagnosis, without a line trigger. It is the same
# pipeline, decision, archive and dispatch a triggered part gets -- this
# route only supplies the frame.
# =============================================================================

inspect_router = APIRouter(prefix="/api/v1", tags=["inspect"])

#: Content types accepted for an uploaded sample.
ALLOWED_IMAGE_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/x-ms-bmp"}
)

#: Filename suffixes accepted when a client sends no usable content type.
ALLOWED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})

#: Largest sample accepted, in bytes. A console upload is one frame, not a
#: dataset; anything larger is a mistake worth reporting as one.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def get_manual_inspector() -> ManualInspectionService:
    """Return the service backing manual ingestion.

    The concrete service is injected via a dependency override at the
    composition root.
    """
    raise NotImplementedError


def _decode_upload(data: bytes) -> Image:
    """Decode uploaded bytes into an RGB image array.

    Args:
        data: Raw file bytes.

    Returns:
        An ``(H, W, 3)`` RGB array.

    Raises:
        HTTPException: If the bytes are not a decodable image.
    """
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be decoded as an image",
        )
    converted: Image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return converted


def _validate_upload_name(file: UploadFile) -> None:
    """Reject uploads that are not one of the accepted image types.

    Args:
        file: The uploaded file.

    Raises:
        HTTPException: If the content type and suffix are both unacceptable.
    """
    content_type = (file.content_type or "").lower()
    if content_type in ALLOWED_IMAGE_TYPES:
        return
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        return
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=f"Unsupported image type: {file.content_type or suffix or 'unknown'}",
    )


@inspect_router.post("/inspect-upload")
async def inspect_upload(
    inspector: Annotated[ManualInspectionService, Depends(get_manual_inspector)],
    file: Annotated[UploadFile, File()],
    part_id: str | None = None,
) -> dict[str, object]:
    """Inspect one uploaded sample through the full pipeline.

    Args:
        inspector: The manual inspection service.
        file: The uploaded image.
        part_id: Identifier to record; defaults to a generated one.

    Returns:
        The inspection outcome, including its measured defect geometry.

    Raises:
        HTTPException: If the upload is too large, of an unsupported type,
            undecodable, or the inspection itself fails.
    """
    _validate_upload_name(file)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    image = _decode_upload(data)

    # Two samples can share a file name (mvtec ships 000.png in every defect
    # directory), so a derived id carries a short token: the MES trail must
    # not show two different parts under one part_id.
    stem = Path(file.filename or "sample").stem or "sample"
    identifier = part_id or f"manual-{stem}-{uuid.uuid4().hex[:6]}"
    try:
        # The pipeline is synchronous and CPU-bound; running it in a worker
        # thread keeps the event loop free for other requests.
        result = await run_in_threadpool(inspector.inspect, image, identifier)
    except AdaptiveVisionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inspection failed: {exc.message}",
        ) from exc

    return {
        "inspection_id": result.inspection_id,
        "part_id": result.part_id,
        "lot_id": result.lot_id,
        "verdict": result.verdict.value,
        "anomaly_score": result.anomaly_score,
        "cycle_time_ms": result.cycle_time_ms,
        "heatmap_region": result.heatmap_region,
        "image_refs": list(result.image_refs),
        "metrology": [m.to_dict() for m in result.defect_measurements],
        "defects": [d.to_dict() for d in result.defects],
        "timestamp_utc": result.timestamp_utc.isoformat(),
    }


# =============================================================================
# Application factory
# =============================================================================

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AdaptiveVision Fab Quality Console</title>
  <style>
    :root {
      --bg: #0e1116; --panel: #161b22; --panel-2: #1c2230; --line: #2a3140;
      --ink: #e6edf3; --ink-dim: #8b949e; --accent: #35c4f0;
      --pass: #3fb950; --fail: #f85149; --review: #d29922; --grid: #222937;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--ink);
      font: 14px/1.5 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    }
    header {
      display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;
      padding: 1rem 1.5rem; border-bottom: 1px solid var(--line); background: var(--panel);
    }
    header h1 { font-size: 1.05rem; margin: 0; letter-spacing: .02em; }
    header .sub { color: var(--ink-dim); font-size: .8rem; }
    header .live { margin-left: auto; color: var(--ink-dim); font-size: .78rem; }
    main { padding: 1.25rem 1.5rem 3rem; display: grid; gap: 1.25rem; }
    .kpis {
      display: grid; gap: .85rem;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }
    .kpi {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; padding: .85rem 1rem;
    }
    .kpi .label {
      color: var(--ink-dim); font-size: .72rem;
      text-transform: uppercase; letter-spacing: .07em;
    }
    .kpi .value {
      font-size: 1.75rem; font-weight: 650; margin-top: .3rem;
      font-variant-numeric: tabular-nums;
    }
    .kpi .note { color: var(--ink-dim); font-size: .72rem; margin-top: .15rem; }
    .kpi.na .value { color: var(--ink-dim); font-size: 1rem; font-weight: 500; padding: .45rem 0; }
    .cols {
      display: grid; gap: 1.25rem;
      grid-template-columns: minmax(340px, 1fr) minmax(340px, 1fr);
    }
    @media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
    section.card {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; overflow: hidden;
    }
    section.card > h2 {
      margin: 0; padding: .7rem 1rem; font-size: .78rem; text-transform: uppercase;
      letter-spacing: .07em; color: var(--ink-dim); border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    .body { padding: 1rem; }
    .pareto { display: grid; gap: .55rem; }
    .bar-row {
      display: grid; grid-template-columns: 132px 1fr 74px;
      gap: .6rem; align-items: center;
    }
    .bar-track { background: var(--grid); border-radius: 3px; height: 15px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--accent); }
    .bar-num {
      text-align: right; color: var(--ink-dim); font-size: .78rem;
      font-variant-numeric: tabular-nums;
    }
    .scroll { overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; font-size: .82rem; }
    th, td {
      padding: .45rem .6rem; text-align: left;
      border-bottom: 1px solid var(--line); white-space: nowrap;
    }
    th {
      color: var(--ink-dim); font-weight: 550; font-size: .72rem;
      text-transform: uppercase; letter-spacing: .05em;
    }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    tbody tr:hover { background: var(--panel-2); cursor: pointer; }
    tbody tr.sel { background: #22304a; }
    .pill {
      padding: .1rem .45rem; border-radius: 999px; font-size: .7rem;
      font-weight: 650; letter-spacing: .04em;
    }
    .pass { background: rgba(63,185,80,.16); color: var(--pass); }
    .fail { background: rgba(248,81,73,.16); color: var(--fail); }
    .review { background: rgba(210,153,34,.16); color: var(--review); }
    .empty { color: var(--ink-dim); font-style: italic; padding: .35rem 0; }
    .adv-row {
      display: flex; gap: .5rem; margin-bottom: .5rem;
      align-items: center; flex-wrap: wrap;
    }
    .adv-h {
      color: var(--ink-dim); font-size: .72rem; text-transform: uppercase;
      letter-spacing: .06em; margin: .8rem 0 .25rem;
    }
    .adv-actions { margin: 0; padding-left: 1.1rem; }
    .adv-actions li { margin-bottom: .25rem; }
    .tag {
      background: var(--grid); color: var(--ink-dim);
      padding: .1rem .45rem; border-radius: 4px; font-size: .7rem;
    }
    .note-box { color: var(--ink-dim); font-size: .78rem; line-height: 1.55; }
    code { background: var(--grid); padding: .05rem .3rem; border-radius: 3px; font-size: .76rem; }
    .viewer-grid {
      display: grid; gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    .tile { margin: 0; }
    .tile img {
      width: 100%; border: 1px solid var(--line); border-radius: 6px;
      background: var(--grid); display: block;
    }
    .tile figcaption { color: var(--ink-dim); font-size: .74rem; margin-top: .35rem; }
    .teach-grid {
      display: grid; gap: 1rem; align-items: start;
      grid-template-columns: minmax(260px, 1fr) minmax(220px, 1fr);
    }
    @media (max-width: 760px) { .teach-grid { grid-template-columns: 1fr; } }
    .drop {
      border: 1.5px dashed var(--line); border-radius: 8px; padding: 1.4rem 1rem;
      text-align: center; color: var(--ink-dim); background: var(--panel-2);
      transition: border-color .15s, background .15s;
    }
    .drop.over { border-color: var(--accent); background: #1a2636; color: var(--ink); }
    .drop strong { color: var(--ink); font-weight: 600; }
    .drop-hint { font-size: .76rem; margin-top: .3rem; }
    .btn-row { display: flex; gap: .5rem; margin-top: .75rem; flex-wrap: wrap; }
    button {
      font: inherit; font-size: .82rem; padding: .42rem .9rem; border-radius: 6px;
      border: 1px solid var(--line); background: var(--panel-2); color: var(--ink);
      cursor: pointer;
    }
    button:hover:not(:disabled) { border-color: var(--accent); }
    button.primary {
      background: var(--accent); border-color: var(--accent);
      color: #06202b; font-weight: 600;
    }
    button:disabled { opacity: .5; cursor: not-allowed; }
    .spinner {
      width: 13px; height: 13px; border: 2px solid var(--line);
      border-top-color: var(--accent); border-radius: 50%;
      display: inline-block; vertical-align: -2px; margin-right: .4rem;
      animation: spin .7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (prefers-reduced-motion: reduce) { .spinner { animation-duration: 2.4s; } }
    .teach-status { margin-top: .7rem; font-size: .82rem; min-height: 1.2rem; }
    .teach-preview img {
      width: 100%; max-height: 190px; object-fit: contain; background: var(--grid);
      border: 1px solid var(--line); border-radius: 6px; display: block;
    }
    .teach-preview figcaption { color: var(--ink-dim); font-size: .74rem; margin-top: .35rem; }
    .err { color: var(--fail); }
  </style>
</head>
<body>
<header>
  <h1>AdaptiveVision &mdash; Fab Quality Console</h1>
  <span class="sub">Deterministic AOI &middot; on-premise inference &middot; local advisory</span>
  <span class="live" id="live">loading&hellip;</span>
</header>

<main>
  <div class="kpis" id="kpis"></div>

  <section class="card" id="teach-card" hidden>
    <h2>Manual Ingestion &mdash; Teach Mode (&#35519;&#27231;&#28204;&#35430;)</h2>
    <div class="body">
      <div class="teach-grid">
        <div>
          <div class="drop" id="drop">
            <strong>Drop a sample image here</strong>
            <div class="drop-hint">PNG, JPG or BMP &middot; or pick a file below</div>
          </div>
          <div class="btn-row">
            <button type="button" id="pick">Choose Sample Image</button>
            <button type="button" id="run" class="primary" disabled>Inspect Sample</button>
          </div>
          <input type="file" id="picker" accept="image/png,image/jpeg,image/bmp" hidden>
          <div class="teach-status" id="teach-status"></div>
        </div>
        <figure class="teach-preview" id="teach-preview" hidden>
          <img id="teach-img" alt="Selected sample">
          <figcaption id="teach-name"></figcaption>
        </figure>
      </div>
      <div class="note-box" style="margin-top:.8rem">
        A submitted sample takes the same path a triggered part does &mdash; same
        preprocessing, model, metrology, deterministic decision, archive and
        MES record. It does not interrupt the running station.
      </div>
    </div>
  </section>

  <div class="cols">
    <section class="card">
      <h2>Defect Pareto</h2>
      <div class="body"><div class="pareto" id="pareto"></div></div>
    </section>
    <section class="card">
      <h2>Advisory (local LLM, explanatory only)</h2>
      <div class="body" id="advisory">
        <div class="empty">Select a row in the MES audit log.</div>
      </div>
    </section>
  </div>

  <section class="card">
    <h2>Review Queue &mdash; parts routed to REVIEW</h2>
    <div class="body scroll" id="reviewq"></div>
  </section>

  <section class="card">
    <h2>MES Audit Log</h2>
    <div class="scroll">
      <table id="log">
        <thead><tr>
          <th>Timestamp (UTC)</th><th>Lot</th><th>Inspection</th><th>Part</th><th>Verdict</th>
          <th class="num">Anomaly</th><th class="num">Largest defect</th>
          <th class="num">Cycle</th><th>Recipe</th><th>Model</th><th>Drift</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>Inspection Viewer</h2>
    <div class="body" id="viewer">
      <div class="empty">Select a row in the MES audit log.</div>
    </div>
  </section>

  <section class="card">
    <h2>Not Connected On This Station</h2>
    <div class="body note-box" id="gaps"></div>
  </section>
</main>

<script>
// ---------------------------------------------------------------------------
// The dashboard derives every number it shows from /api/v1/results, which is
// the persisted traceability record itself. Nothing here is estimated: a
// metric that needs a data feed this station does not have (an upstream
// inspection verdict, a downstream audit result) is reported as unavailable
// rather than shown as a plausible-looking number.
// ---------------------------------------------------------------------------
var STATE = { items: [] };

function pct(n) { return (n * 100).toFixed(2) + '%'; }
function pill(v) { return '<span class="pill ' + v + '">' + v.toUpperCase() + '</span>'; }
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}
function quantile(sorted, q) {
  if (!sorted.length) { return null; }
  var i = Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1);
  return sorted[Math.max(0, i)];
}
function fmtScore(v) { return v == null ? '&mdash;' : v.toFixed(3); }
function largestDefectUm2(item) {
  var ms = item.defect_measurements || [];
  var best = 0;
  ms.forEach(function (m) { if (m.area_um2 > best) { best = m.area_um2; } });
  return best;
}

function kpiCard(label, value, note) {
  return '<div class="kpi"><div class="label">' + label + '</div>' +
    '<div class="value">' + value + '</div>' +
    (note ? '<div class="note">' + note + '</div>' : '') + '</div>';
}
function kpiNA(label, why) {
  return '<div class="kpi na"><div class="label">' + label + '</div>' +
    '<div class="value">not connected</div><div class="note">' + why + '</div></div>';
}

function renderKpis() {
  var items = STATE.items;
  var total = items.length;
  var passed = items.filter(function (i) { return i.verdict === 'pass'; }).length;
  var review = items.filter(function (i) { return i.verdict === 'review'; }).length;
  var cycles = items.map(function (i) { return i.cycle_time_ms; })
                    .filter(function (c) { return typeof c === 'number'; })
                    .sort(function (a, b) { return a - b; });
  var mean = cycles.length
    ? cycles.reduce(function (a, b) { return a + b; }, 0) / cycles.length : null;
  var p95 = quantile(cycles, 0.95);
  var uph = mean ? Math.round(3600000 / mean) : null;

  var html = '';
  html += kpiCard('Line Yield', total ? pct(passed / total) : '&mdash;',
                  passed + ' pass / ' + total + ' inspected');
  html += kpiCard('Throughput (UPH)', uph == null ? '&mdash;' : uph.toLocaleString(),
                  mean == null ? '' : 'mean cycle ' + mean.toFixed(1) + ' ms');
  html += kpiCard('P95 Cycle', p95 == null ? '&mdash;' : p95.toFixed(1) + ' ms',
                  '30 ms takt budget');
  html += kpiCard('Review Queue', String(review), 'awaiting human re-judge');
  html += kpiNA('Overkill Reduction', 'needs an upstream inspection verdict feed');
  html += kpiNA('Escape Rate', 'needs a downstream audit / final-test feed');
  document.getElementById('kpis').innerHTML = html;
}

function renderPareto() {
  var counts = {};
  STATE.items.forEach(function (item) {
    (item.defects || []).forEach(function (d) {
      counts[d.defect_class] = (counts[d.defect_class] || 0) + 1;
    });
    (item.defect_measurements || []).forEach(function (m) {
      if (m.morphology) {
        var k = 'morphology: ' + m.morphology;
        counts[k] = (counts[k] || 0) + 1;
      }
    });
  });
  var rows = Object.keys(counts).map(function (k) { return [k, counts[k]]; })
                   .sort(function (a, b) { return b[1] - a[1]; });
  var el = document.getElementById('pareto');
  if (!rows.length) {
    el.innerHTML = '<div class="empty">No defects recorded in this window.</div>';
    return;
  }
  var total = rows.reduce(function (a, r) { return a + r[1]; }, 0);
  var max = rows[0][1];
  el.innerHTML = rows.map(function (r) {
    return '<div class="bar-row"><span>' + esc(r[0]) + '</span>' +
      '<span class="bar-track"><span class="bar-fill" style="width:' +
      (100 * r[1] / max).toFixed(1) + '%"></span></span>' +
      '<span class="bar-num">' + r[1] + ' &middot; ' +
      (100 * r[1] / total).toFixed(0) + '%</span></div>';
  }).join('');
}

function renderReviewQueue() {
  var q = STATE.items.filter(function (i) { return i.verdict === 'review'; });
  var el = document.getElementById('reviewq');
  if (!q.length) {
    el.innerHTML = '<div class="empty">Empty &mdash; no parts routed to REVIEW.</div>';
    return;
  }
  el.innerHTML = '<table><thead><tr><th>Inspection</th><th>Part</th>' +
    '<th class="num">Anomaly</th><th class="num">Largest defect</th><th>Timestamp</th>' +
    '</tr></thead><tbody>' + q.map(function (i) {
      return '<tr><td>' + esc(i.inspection_id) + '</td><td>' + esc(i.part_id) + '</td>' +
        '<td class="num">' + fmtScore(i.anomaly_score) + '</td>' +
        '<td class="num">' + largestDefectUm2(i).toFixed(0) + ' &micro;m&sup2;</td>' +
        '<td>' + esc(i.timestamp_utc) + '</td></tr>';
    }).join('') + '</tbody></table>' +
    '<div class="note-box" style="margin-top:.6rem">Read-only: recording an ' +
    'operator disposition needs a write endpoint, which this API does not expose.</div>';
}

function renderLog() {
  var body = document.querySelector('#log tbody');
  if (!STATE.items.length) {
    body.innerHTML = '<tr><td colspan="11" class="empty">No inspections persisted yet. ' +
      'Run <code>python scripts/run_station.py</code> to produce records.</td></tr>';
    return;
  }
  body.innerHTML = STATE.items.map(function (i, idx) {
    var um2 = largestDefectUm2(i);
    return '<tr data-idx="' + idx + '">' +
      '<td>' + esc(i.timestamp_utc) + '</td>' +
      '<td>' + esc(i.lot_id) + '</td>' +
      '<td>' + esc(i.inspection_id) + '</td>' +
      '<td>' + esc(i.part_id) + '</td>' +
      '<td>' + pill(i.verdict) + '</td>' +
      '<td class="num">' + fmtScore(i.anomaly_score) + '</td>' +
      '<td class="num">' + (um2 ? um2.toFixed(0) + ' &micro;m&sup2;' : '&mdash;') + '</td>' +
      '<td class="num">' + i.cycle_time_ms.toFixed(1) + ' ms</td>' +
      '<td>' + esc(i.recipe_ver) + '</td>' +
      '<td>' + esc(i.model_ver) + '</td>' +
      '<td>' + esc(i.drift_status || '&mdash;') + '</td></tr>';
  }).join('');

  Array.prototype.forEach.call(body.querySelectorAll('tr[data-idx]'), function (tr) {
    tr.addEventListener('click', function () {
      Array.prototype.forEach.call(body.querySelectorAll('tr'), function (r) {
        r.classList.remove('sel');
      });
      tr.classList.add('sel');
      var picked = STATE.items[Number(tr.getAttribute('data-idx'))];
      loadAdvisory(picked);
      loadViewer(picked);
    });
  });
}

// Advisory reports are fetched per selected row, best-effort: a 404 (no
// report yet, or no advisory repository configured on this station) renders
// as an explanatory note rather than an error.
function loadAdvisory(item) {
  var el = document.getElementById('advisory');
  el.innerHTML = '<div class="empty">Loading advisory for ' +
    esc(item.inspection_id) + '&hellip;</div>';
  fetch('/api/v1/advisory/' + encodeURIComponent(item.inspection_id))
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (rep) {
      if (!rep) {
        el.innerHTML = '<div class="note-box">No advisory report stored for <code>' +
          esc(item.inspection_id) + '</code>.<br>Reports are produced by the local ' +
          'Ollama engine and persisted separately; the deterministic verdict above ' +
          'does not depend on them.</div>';
        return;
      }
      var acts = (rep.recommended_actions || []).map(function (a) {
        return '<li>' + esc(a) + '</li>';
      }).join('');
      el.innerHTML =
        '<div class="adv-row">' + pill(item.verdict) +
          '<span class="tag">severity: ' + esc(rep.severity) + ' (deterministic)</span>' +
          '<span class="tag">class: ' + esc(rep.defect_classification) + '</span>' +
          '<span class="tag">confidence: ' + Number(rep.confidence_score).toFixed(2) + '</span>' +
          (rep.is_fallback ? '<span class="tag">rule-based fallback</span>'
                           : '<span class="tag">local LLM</span>') +
        '</div>' +
        '<div class="adv-h">Root-cause hypothesis</div>' +
        '<div>' + esc(rep.root_cause_hypothesis) + '</div>' +
        (acts ? '<div class="adv-h">Technician SOP</div>' +
                '<ol class="adv-actions">' + acts + '</ol>' : '');
    })
    .catch(function () {
      el.innerHTML = '<div class="note-box">Advisory service unreachable.</div>';
    });
}

// The station archives the frame it actually inspected. A reference that
// is a bare frame id (no archive configured when the part ran) has no image
// behind it, and is reported as such rather than rendered as a broken tile.
function loadViewer(item) {
  var el = document.getElementById('viewer');
  var refs = item.image_refs || [];
  if (!refs.length) {
    el.innerHTML = '<div class="empty">No image references on this record.</div>';
    return;
  }
  var tiles = refs.map(function (ref, idx) {
    var id = String(ref).split('/').pop().replace(/\\.bin$/, '');
    var label = idx === 0 ? 'Raw inspected frame' : 'Anomaly heatmap';
    return '<figure class="tile">' +
      '<img src="/api/v1/images/' + encodeURIComponent(id) + '" alt="' + label + '" ' +
        'onerror="this.replaceWith(Object.assign(document.createElement(\'div\'),' +
        '{className:\'empty\',textContent:\'not archived\'}))">' +
      '<figcaption>' + label + ' &middot; <span class="tag">' + esc(id) + '</span></figcaption>' +
      '</figure>';
  }).join('');
  el.innerHTML = '<div class="viewer-grid">' + tiles + '</div>' +
    (refs.length === 1
      ? '<div class="note-box" style="margin-top:.6rem">Only the raw frame is ' +
        'archived: the configured detector reports a scalar score and no ' +
        'localization map, so there is no heatmap to show beside it.</div>'
      : '');
}

function renderGaps() {
  document.getElementById('gaps').innerHTML =
    '<b>Anomaly heatmap</b> &mdash; the raw frame is archived and shown ' +
    'above, but the exported model returns a scalar score only (its patch ' +
    'memory bank is internal to the graph), so there is no per-patch map to ' +
    'display beside it. It needs a re-export that emits per-patch distances, ' +
    'or a normal-patch reference bank to measure against.<br><br>' +
    '<b>Overkill reduction / escape rate</b> &mdash; both compare this ' +
    'station against another measurement (an upstream inspector, a ' +
    'downstream audit). Neither feed exists here, so neither number is shown.<br><br>' +
    '<b>Recipe changeover</b> &mdash; recipes load from disk at boot ' +
    '(<code>JsonRecipeStore</code>); switching one from this page needs a ' +
    'write endpoint that is not exposed.';
}

// --- Teach mode -----------------------------------------------------------
// The panel is shown only when the station exposed the ingestion route; a
// read-only dashboard has no way to run an inspection and should not offer
// a control that cannot work.
var SELECTED_FILE = null;

function teachStatus(html, isError) {
  var el = document.getElementById('teach-status');
  el.innerHTML = html;
  el.className = 'teach-status' + (isError ? ' err' : '');
}

function selectFile(file) {
  if (!file) { return; }
  SELECTED_FILE = file;
  document.getElementById('run').disabled = false;
  document.getElementById('teach-name').textContent = file.name;
  var preview = document.getElementById('teach-preview');
  document.getElementById('teach-img').src = URL.createObjectURL(file);
  preview.hidden = false;
  teachStatus('Ready to inspect ' + esc(file.name) + '.');
}

function runInspection() {
  if (!SELECTED_FILE) { return; }
  var button = document.getElementById('run');
  button.disabled = true;
  teachStatus('<span class="spinner"></span>Inspecting&hellip;');

  var body = new FormData();
  body.append('file', SELECTED_FILE);
  fetch('/api/v1/inspect-upload', { method: 'POST', body: body })
    .then(function (r) {
      return r.json().then(function (payload) {
        if (!r.ok) { throw new Error(payload.detail || ('HTTP ' + r.status)); }
        return payload;
      });
    })
    .then(function (outcome) {
      var geometry = (outcome.metrology || []).map(function (m) {
        return m.morphology + ' ' + Math.round(m.area_um2) + ' \u00b5m\u00b2';
      }).join(', ');
      teachStatus(pill(outcome.verdict) +
        ' &nbsp;score ' + (outcome.anomaly_score == null
          ? '&mdash;' : Number(outcome.anomaly_score).toFixed(4)) +
        ' &middot; ' + Number(outcome.cycle_time_ms).toFixed(0) + ' ms' +
        (geometry ? ' &middot; ' + esc(geometry) : ''));
      // Refresh from the database rather than splicing a local object in:
      // the row the operator inspects is then the persisted record itself.
      return refresh().then(function () { selectRowById(outcome.inspection_id); });
    })
    .catch(function (err) {
      teachStatus('Inspection failed: ' + esc(err.message), true);
    })
    .finally(function () { button.disabled = false; });
}

function selectRowById(inspectionId) {
  var index = STATE.items.findIndex(function (i) {
    return i.inspection_id === inspectionId;
  });
  if (index < 0) { return; }
  var row = document.querySelector('#log tbody tr[data-idx="' + index + '"]');
  if (row) { row.click(); row.scrollIntoView({ block: 'nearest' }); }
}

function initTeachMode() {
  // HEAD tells us whether the route exists without running an inspection.
  fetch('/api/v1/inspect-upload', { method: 'POST' })
    .then(function (r) {
      if (r.status === 404) { return; }
      document.getElementById('teach-card').hidden = false;
      wireTeachMode();
    })
    .catch(function () { /* leave the panel hidden */ });
}

function wireTeachMode() {
  var drop = document.getElementById('drop');
  var picker = document.getElementById('picker');

  document.getElementById('pick').addEventListener('click', function () { picker.click(); });
  picker.addEventListener('change', function () { selectFile(picker.files[0]); });
  document.getElementById('run').addEventListener('click', runInspection);

  ['dragenter', 'dragover'].forEach(function (name) {
    drop.addEventListener(name, function (e) {
      e.preventDefault(); drop.classList.add('over');
    });
  });
  ['dragleave', 'drop'].forEach(function (name) {
    drop.addEventListener(name, function (e) {
      e.preventDefault(); drop.classList.remove('over');
    });
  });
  drop.addEventListener('drop', function (e) {
    if (e.dataTransfer.files.length) { selectFile(e.dataTransfer.files[0]); }
  });
}

function refresh() {
  return fetch('/api/v1/results?limit=200')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      STATE.items = data.items || [];
      document.getElementById('live').textContent =
        STATE.items.length + ' records \u00b7 updated ' + new Date().toLocaleTimeString();
      renderKpis(); renderPareto(); renderReviewQueue(); renderLog(); renderGaps();
    })
    .catch(function () {
      document.getElementById('live').textContent = 'API unreachable';
    });
}

refresh();
initTeachMode();
setInterval(refresh, 5000);
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
    image_store: LocalImageStore | None = None,
    manual_inspector: ManualInspectionService | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        repository: The result repository backing the results endpoints.
        metrics: The metrics registry backing the metrics endpoints.
        health: The health check backing the health endpoints.
        advisory: The advisory repository backing the advisory endpoints
            (Milestone M19). When ``None`` (the default), the advisory routes
            are not registered at all - a fully supported configuration.
        manual_inspector: Service backing ``POST /api/v1/inspect-upload``.
            When ``None`` (the default), that route is not registered and the
            console's teach-mode panel is hidden -- a read-only dashboard.
        image_store: The archive backing ``/api/v1/images`` (Milestone M22).
            When ``None`` (the default), the image routes are not registered
            at all - the same optional-subsystem pattern as ``advisory``.
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

    if image_store is not None:

        def _get_image_store() -> LocalImageStore:
            return image_store

        app.dependency_overrides[get_image_store] = _get_image_store
        app.include_router(images_router)

    if manual_inspector is not None:

        def _get_manual_inspector() -> ManualInspectionService:
            return manual_inspector

        app.dependency_overrides[get_manual_inspector] = _get_manual_inspector
        app.include_router(inspect_router)

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
