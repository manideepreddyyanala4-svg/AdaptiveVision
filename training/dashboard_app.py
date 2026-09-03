"""A standalone visual dashboard for the trained PaDiM anomaly models.

This is deliberately separate from the production dashboard
(``adaptivevision.api`` / ``scripts/run_api.py``, which lists raw JSON
inspection rows with no images) -- it exists to actually *see* results: real
thumbnails in a filterable gallery with a live confusion matrix, and a
drag-and-drop box to upload one or many images and run them through a chosen
model immediately. Every score shown here comes from the project's own
``OnnxInferenceEngine`` + ``ThresholdAnomalyDetector``; nothing is
reimplemented or faked.

The `adaptivevision.db` this reads from is shared with the rest of the
project (walking-skeleton demo runs, earlier sessions, etc.), so every query
here explicitly scopes to rows this dashboard itself wrote (``station_id``
prefixed ``demo-``, produced by ``push_results_to_dashboard.py``) and treats
every field defensively -- other rows in that table were never written by
this code and may have a null ``anomaly_score`` or a different shape.

Usage:
    python training/dashboard_app.py
    # open http://127.0.0.1:8010/
"""

from __future__ import annotations

import base64
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from adaptivevision.explanation import OllamaAdvisoryEngine
from adaptivevision.explanation import advise, build_evidence
from adaptivevision.common import ExecutionProvider, Verdict
from adaptivevision.common import DefectMeasurement
from adaptivevision.common import RectifiedFrame
from adaptivevision.config import load_aoi_config
from adaptivevision.decision import Decision
from adaptivevision.engine import OnnxInferenceEngine
from adaptivevision.metrology import ThresholdAnomalyDetector
from adaptivevision.metrology import MetrologyConfig, measure_defects
from adaptivevision.drift import DriftDetector, DriftReport
from adaptivevision.storage import open_database
from adaptivevision.storage import SqliteResultRepository
from adaptivevision.explanation import FaissRetrievalIndex

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DB_PATH = REPO_ROOT / "adaptivevision.db"
THUMBNAIL_MAX_SIDE = 260
#: Only rows written by push_results_to_dashboard.py carry this station_id
#: prefix; everything else in the shared DB (other milestones, other
#: sessions) is deliberately excluded from every query below.
GALLERY_STATION_PREFIX = "demo-"

app = FastAPI(title="AdaptiveVision PaDiM Dashboard")

_cache_lock = threading.Lock()
_detector_cache: dict[
    str, tuple[ThresholdAnomalyDetector, dict[str, Any], OnnxInferenceEngine]
] = {}
_thumbnail_cache: dict[str, str] = {}

_, _session_factory = open_database(str(DB_PATH))
_repository = SqliteResultRepository(_session_factory)

#: Milestone M19: advisory (root-cause explanation) engine. Construction is
#: cheap and side-effect-free (no network call, no server connection) - it
#: only lazily imports/calls Ollama inside ``generate_report``, falling back
#: deterministically if it is unavailable. There is no FAISS retrieval wired
#: into this dashboard for PaDiM models: their ONNX contract predates the
#: embedding output and still emits only a calibrated scalar score. Models
#: exported after 2026-08-27 (see ``export.py``'s ``ProductionExport``) emit
#: an ``"embedding"`` output too; :func:`_retrieval_index_for` loads a FAISS
#: index for any such model that has one built (see
#: ``build_retrieval_index.py``), and retrieval is skipped, not faked, for
#: every model that does not.
_advisory_engine = OllamaAdvisoryEngine(model="llama3:latest")

RETRIEVAL_DIR = REPO_ROOT / "training" / "benchmark_results" / "retrieval"
_retrieval_index_cache: dict[str, FaissRetrievalIndex | None] = {}

#: Milestone M21: metrology calibration + drift thresholds, loaded once from
#: configs/config.yaml (see adaptivevision.config).
_aoi_config = load_aoi_config()

#: Real scores bootstrapped so far per model, before a DriftDetector exists.
_drift_baselines: dict[str, list[float]] = {}
#: One DriftDetector per model, built once its baseline is fully bootstrapped.
_drift_detectors: dict[str, DriftDetector] = {}
#: Scores needed to bootstrap a model's baseline before drift checks start.
#: Deliberately real observed scores, not synthesized ones: an earlier
#: version of this function drew a synthetic Normal(calibration_mu,
#: calibration_sigma) baseline from the manifest, but calibration_mu/sigma
#: are the *pre-sigmoid* raw-score statistics computed at export time (see
#: training/benchmark/export.py's ProductionExport, which bakes
#: sigmoid((raw - mu) / sigma) into the graph itself) -- a completely
#: different scale from the exported model's actual (post-sigmoid) output
#: score. Comparing that output against a baseline built from the
#: pre-sigmoid statistic is a unit mismatch, not a real drift signal, and
#: produced a false SENSOR_DRIFT_ALERT on the very second upload during
#: testing. This project has no persisted store of real historical
#: known-good scores to seed a baseline from instead, so this bootstraps one
#: from whatever's actually uploaded -- an honest baseline on the right
#: scale, at the cost of assuming early uploads are representative of normal
#: operation (a real deployment should instead seed this from a curated
#: batch of confirmed-good commissioning images, not live traffic that may
#: itself include defective parts).
_DRIFT_BASELINE_SIZE = 20


def _drift_report_for(manifest: dict[str, Any], score: float) -> DriftReport | None:
    """Update this model's drift baseline/window with ``score``.

    Args:
        manifest: The manifest of the model that produced ``score``.
        score: The anomaly score just computed for one inspection.

    Returns:
        A :class:`DriftReport` once this model has a fully bootstrapped
        baseline (see :data:`_DRIFT_BASELINE_SIZE`) and at least two scores
        recorded against it, otherwise ``None`` -- including for the score
        that completes the baseline itself, since it was just absorbed into
        the reference distribution rather than compared against it.
    """
    name = manifest.get("name", "")
    if not name:
        return None

    if name not in _drift_detectors:
        baseline = _drift_baselines.setdefault(name, [])
        baseline.append(score)
        if len(baseline) < _DRIFT_BASELINE_SIZE:
            return None
        _drift_detectors[name] = DriftDetector(
            baseline,
            window_size=_aoi_config.drift.window_size,
            p_value_threshold=_aoi_config.drift.p_value_threshold,
        )
        return None

    return _drift_detectors[name].record(score)


def _retrieval_index_for(model_name: str) -> FaissRetrievalIndex | None:
    """Return the cached FAISS index for ``model_name``, or ``None`` if it has none."""
    if model_name in _retrieval_index_cache:
        return _retrieval_index_cache[model_name]
    index_path = RETRIEVAL_DIR / (Path(model_name).stem + ".faiss")
    index: FaissRetrievalIndex | None = None
    if index_path.exists():
        try:
            sidecar = json.loads((RETRIEVAL_DIR / (index_path.name + ".meta.json")).read_text())
            meta = sidecar["index_metadata"]
            index = FaissRetrievalIndex(
                meta["dim"],
                metric=meta["metric"],
                embedding_model=meta["embedding_model"],
                embedding_version=meta["embedding_version"],
                preprocessing_version=meta["preprocessing_version"],
            )
            index.load(index_path)
        except Exception as exc:  # noqa: BLE001 - a bad index must not break scoring
            print(f"failed to load retrieval index for {model_name}: {exc}")
            index = None
    _retrieval_index_cache[model_name] = index
    return index


def _load_manifests() -> list[dict[str, Any]]:
    """List every trained model (any method family) with a valid ``<name>.json`` sidecar."""
    manifests = []
    for onnx_path in sorted(MODELS_DIR.glob("*.onnx")):
        manifest_path = onnx_path.with_suffix(".json")
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["name"] = onnx_path.name
        data.setdefault("test_auroc", None)
        data.setdefault("category", None)
        manifests.append(data)
    return manifests


def _get_detector(
    model_name: str,
) -> tuple[ThresholdAnomalyDetector, dict[str, Any], OnnxInferenceEngine]:
    """Return a cached (loading if needed) detector + manifest + engine for ``model_name``.

    The engine is returned alongside the detector so callers can also read
    its ``"embedding"`` output (Milestone M19 retrieval) without loading the
    model a second time - ``ThresholdAnomalyDetector.detect`` only reads
    ``outputs["output"]``.
    """
    with _cache_lock:
        cached = _detector_cache.get(model_name)
        if cached is not None:
            return cached
        manifest = next((m for m in _load_manifests() if m["name"] == model_name), None)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Unknown model {model_name!r}")
        engine = OnnxInferenceEngine(model_dir=MODELS_DIR, providers=(ExecutionProvider.CPU,))
        try:
            engine.load(model_name)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc
        detector = ThresholdAnomalyDetector(engine, threshold=0.5)
        _detector_cache[model_name] = (detector, manifest, engine)
        return detector, manifest, engine


def _encode_thumbnail(bgr: np.ndarray) -> str:
    """Downscale + JPEG-encode a BGR array into a data: URL for inline display."""
    height, width = bgr.shape[:2]
    scale = min(1.0, THUMBNAIL_MAX_SIDE / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode thumbnail")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _cached_thumbnail(path: Path) -> str | None:
    """Thumbnail a dataset image, cached by (path, mtime) since the gallery re-reads often."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = f"{path}:{mtime}"
    cached = _thumbnail_cache.get(key)
    if cached is not None:
        return cached
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    encoded = _encode_thumbnail(bgr)
    _thumbnail_cache[key] = encoded
    return encoded


#: 3x3 coarse region names, indexed [row][col], row/col in {0,1,2}.
_REGION_NAMES = (
    ("upper-left", "upper-center", "upper-right"),
    ("center-left", "center", "center-right"),
    ("lower-left", "lower-center", "lower-right"),
)


def _compute_heatmap(
    patch_features: np.ndarray, rgb_resized: np.ndarray
) -> tuple[str, str, list[DefectMeasurement]] | tuple[None, None, list[DefectMeasurement]]:
    """Overlay a per-patch anomaly heatmap on the resized query image, and
    measure the defect regions in it.

    The exported ONNX graph's ``patch_features`` output gives each patch's
    feature vector but not its distance to the fitted memory bank (that bank
    isn't part of the exported graph), so this can't reproduce the exact
    per-patch score PatchCore itself uses. Instead it scores each patch by
    its distance from this *same image's* own median patch: real defects are
    the visual outlier against the rest of a mostly-normal part, which is
    what makes this a legitimate (if approximate) localization signal rather
    than a fabricated one -- it degrades gracefully (returns
    ``(None, None, [])``) rather than pretending precision it doesn't have
    when the patch grid can't be reshaped cleanly.

    Defect metrology (Milestone M21) runs on the same upsampled grid the
    color overlay is built from, in real display-pixel units, so
    ``DefectMeasurement.bbox``/``area_px2`` line up with the returned image.

    Args:
        patch_features: ``(n_patches, dim)`` feature vectors from the model.
        rgb_resized: The ``(H, W, 3)`` RGB image the model actually saw.

    Returns:
        ``(heatmap data: URL, coarse region name, defect measurements)``, or
        ``(None, None, [])`` if the patch grid shape can't be recovered.
    """
    n_patches = patch_features.shape[0]
    grid = int(round(n_patches**0.5))
    if grid * grid != n_patches:
        return None, None, []

    median_patch = np.median(patch_features, axis=0, keepdims=True)
    distances = np.linalg.norm(patch_features - median_patch, axis=1)
    grid_map = distances.reshape(grid, grid)

    peak_row, peak_col = np.unravel_index(np.argmax(grid_map), grid_map.shape)
    region = _REGION_NAMES[min(2, peak_row * 3 // grid)][min(2, peak_col * 3 // grid)]

    lo, hi = float(grid_map.min()), float(grid_map.max())
    normalized = np.zeros_like(grid_map, dtype=np.uint8) if hi <= lo else (
        (grid_map - lo) / (hi - lo) * 255
    ).astype(np.uint8)

    height, width = rgb_resized.shape[:2]
    upsampled = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(upsampled, cv2.COLORMAP_TURBO)
    bgr_resized = cv2.cvtColor(rgb_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr_resized, 0.55, colored, 0.45, 0)

    ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None, None, []
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    measurements = measure_defects(
        upsampled.astype(np.float64),
        MetrologyConfig(
            pixel_to_micron=_aoi_config.metrology.pixel_to_micron,
            min_area_px2=_aoi_config.metrology.min_area_px2,
            threshold_percentile=_aoi_config.metrology.threshold_percentile,
        ),
    )
    return data_url, region, measurements


def _score_image(
    bgr: np.ndarray,
    filename: str,
    detector: ThresholdAnomalyDetector,
    manifest: dict[str, Any],
    engine: OnnxInferenceEngine,
) -> dict[str, Any]:
    """Run one decoded BGR image through ``detector`` and shape the API response."""
    # (H, W, 3) channel-last -- ThresholdAnomalyDetector.detect() transposes to
    # the model's (3, H, W) contract itself. Feeding it bgr_to_model_input()'s
    # already-transposed output here double-transposed it (production bug,
    # same root cause already fixed in training/benchmark/export.py's
    # verification step: see docs/milestones/M20.md).
    resized = cv2.resize(bgr, (manifest["width"], manifest["height"]), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    frame = RectifiedFrame(
        image=image,
        camera_id="dashboard-upload",
        frame_id=Path(filename).stem or "upload",
        calibration_ver="n/a",
        timestamp_monotonic=0.0,
        timestamp_utc=datetime.now(UTC),
    )
    result = detector.detect(frame)

    outputs = engine.infer({"input": image.transpose(2, 0, 1)})
    heatmap, heatmap_region, defect_measurements = _compute_heatmap(
        np.asarray(outputs["patch_features"]), image
    )

    drift_report = _drift_report_for(manifest, result.score)
    drift_status = drift_report.status if drift_report is not None else None

    retrieval_matches = ()
    index = _retrieval_index_for(manifest.get("name", ""))
    if index is not None:
        embedding = outputs["embedding"]
        retrieval_matches = index.search(np.asarray(embedding, dtype=np.float32), top_k=3)

    decision = Decision(
        verdict=Verdict.FAIL if result.is_anomalous else Verdict.PASS,
        defects=result.defects,
    )
    evidence = build_evidence(
        inspection_id=frame.frame_id,
        category=manifest.get("category") or "unknown",
        anomaly_score=result.score,
        model_ver=manifest.get("name", "unknown"),
        decision=decision,
        retrieval_matches=retrieval_matches,
        heatmap_region=heatmap_region,
    )
    report = advise(evidence, advisory=_advisory_engine)

    return {
        "filename": filename,
        "score": result.score,
        "threshold": result.threshold,
        "is_anomalous": result.is_anomalous,
        "verdict": "FAIL" if result.is_anomalous else "PASS",
        "defects": [d.description for d in result.defects if d.description],
        "thumbnail": _encode_thumbnail(bgr),
        "heatmap": heatmap,
        "heatmap_region": heatmap_region,
        "resized_to": f"{manifest['width']}x{manifest['height']}",
        "advisory": report.to_dict() if report is not None else None,
        "similar_historical_defects": [m.to_dict() for m in retrieval_matches],
        "defect_measurements": [m.to_dict() for m in defect_measurements],
        "drift_status": drift_status,
        "drift_report": drift_report.to_dict() if drift_report is not None else None,
    }


def _ground_truth_of(image_refs: tuple[str, ...]) -> str | None:
    return next(
        (ref.split("=", 1)[1] for ref in image_refs if ref.startswith("ground_truth=")),
        None,
    )


def _demo_rows() -> list[Any]:
    """All dashboard-written rows (see ``GALLERY_STATION_PREFIX``), newest first."""
    raw = _repository.list_results(limit=2000)
    rows = [
        r
        for r in raw
        if r.station_id.startswith(GALLERY_STATION_PREFIX) and r.anomaly_score is not None
    ]
    rows.sort(key=lambda r: r.timestamp_utc, reverse=True)
    return rows


@app.get("/api/models")
def list_models() -> JSONResponse:
    """List every trained PaDiM model with its manifest metadata."""
    return JSONResponse(_load_manifests())


@app.post("/api/reload")
def reload_models() -> JSONResponse:
    """Clear the loaded-model and thumbnail caches (e.g. after training a new model)."""
    with _cache_lock:
        _detector_cache.clear()
    _thumbnail_cache.clear()
    return JSONResponse({"reloaded": True, "models": len(_load_manifests())})


@app.post("/api/score")
async def score_uploads(
    files: list[UploadFile] = File(...), model: str = Form(...)
) -> JSONResponse:
    """Score one or many uploaded images with the chosen model, in one batch."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    detector, manifest, engine = _get_detector(model)

    results = []
    for upload in files:
        filename = upload.filename or "upload"
        raw_bytes = await upload.read()
        if not raw_bytes:
            results.append({"filename": filename, "error": "empty file"})
            continue
        bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            results.append({"filename": filename, "error": "not a decodable image"})
            continue
        try:
            results.append(_score_image(bgr, filename, detector, manifest, engine))
        except Exception as exc:
            results.append({"filename": filename, "error": str(exc)})

    return JSONResponse(
        {
            "model": model,
            "dataset": manifest.get("dataset"),
            "category": manifest.get("category"),
            "results": results,
        }
    )


@app.get("/api/gallery")
def gallery(
    limit: int = Query(200, ge=1, le=1000),
    dataset: str = "all",
    verdict: str = "all",
    sort: str = "newest",
) -> JSONResponse:
    """Return dashboard-written inspection results with real thumbnails, filtered/sorted."""
    items = []
    for result in _demo_rows():
        result_dataset = result.station_id[len(GALLERY_STATION_PREFIX) :]
        if dataset != "all" and not result_dataset.startswith(dataset):
            continue
        if verdict != "all" and result.verdict.value != verdict:
            continue
        source_path = Path(result.image_refs[0]) if result.image_refs else None
        thumbnail = _cached_thumbnail(source_path) if source_path else None
        items.append(
            {
                "inspection_id": result.inspection_id,
                "station_id": result.station_id,
                "dataset": result_dataset,
                "model_ver": result.model_ver,
                "verdict": result.verdict.value,
                "anomaly_score": result.anomaly_score,
                "ground_truth": _ground_truth_of(result.image_refs),
                "timestamp_utc": result.timestamp_utc.isoformat(),
                "thumbnail": thumbnail,
                "source_name": source_path.name if source_path else None,
            }
        )

    if sort == "score_desc":
        items.sort(key=lambda item: item["anomaly_score"], reverse=True)
    elif sort == "score_asc":
        items.sort(key=lambda item: item["anomaly_score"])
    elif sort == "oldest":
        items.reverse()
    # "newest" is already the incoming order from _demo_rows().

    return JSONResponse({"items": items[:limit], "total_matching": len(items)})


@app.get("/api/stats")
def stats() -> JSONResponse:
    """Live confusion-matrix stats per model, computed from the saved demo rows."""
    by_model: dict[str, dict[str, int]] = {}
    for result in _demo_rows():
        ground_truth = _ground_truth_of(result.image_refs)
        if ground_truth is None:
            continue
        bucket = by_model.setdefault(
            result.model_ver,
            {
                "total": 0,
                "correct": 0,
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
            },
        )
        predicted_anomaly = result.verdict.value == "fail"
        actual_anomaly = ground_truth == "anomaly"
        bucket["total"] += 1
        bucket["correct"] += int(predicted_anomaly == actual_anomaly)
        if predicted_anomaly and actual_anomaly:
            bucket["true_positive"] += 1
        elif not predicted_anomaly and not actual_anomaly:
            bucket["true_negative"] += 1
        elif predicted_anomaly and not actual_anomaly:
            bucket["false_positive"] += 1
        else:
            bucket["false_negative"] += 1
    return JSONResponse(by_model)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Avoid noisy 404s in the server log for the browser's automatic favicon request."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the dashboard page."""
    return _PAGE_HTML


_PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AdaptiveVision - PaDiM Anomaly Dashboard</title>
<style>
  :root {
    --bg: #0b0f14; --panel: #121821; --panel2: #1a2230; --border: #26303f;
    --text: #e6edf3; --muted: #93a3b8; --accent: #4f9dff;
    --pass: #2ecc71; --fail: #ff5d5d; --warn: #e8b339;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--bg); color: var(--text);
  }
  header { padding: 20px 32px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0; font-size: 20px; }
  header p { margin: 4px 0 12px; color: var(--muted); font-size: 13px; }
  #stat-chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    font-size: 12px; padding: 5px 10px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--panel2); color: var(--muted);
  }
  .chip b { color: var(--text); }
  .chip.good { border-color: rgba(46,204,113,.5); }
  .chip.mid { border-color: rgba(232,179,57,.5); }
  .chip.bad { border-color: rgba(255,93,93,.5); }
  main { max-width: 1180px; margin: 0 auto; padding: 28px 32px 60px; }
  section { margin-bottom: 40px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-bottom: 14px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
  .upload-row { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
  #dropzone {
    flex: 1 1 340px; min-height: 160px; border: 2px dashed var(--border); border-radius: 10px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    color: var(--muted); cursor: pointer; transition: border-color .15s, background .15s;
    text-align: center; padding: 16px; gap: 10px;
  }
  #dropzone.drag { border-color: var(--accent); background: #132033; }
  #file-strip { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
  #file-strip .thumb { position: relative; width: 56px; height: 56px; border-radius: 6px; overflow: hidden; }
  #file-strip .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  #file-strip .thumb button {
    position: absolute; top: 0; right: 0; width: 18px; height: 18px; padding: 0; line-height: 1;
    border-radius: 0 0 0 6px; background: rgba(0,0,0,.6); color: #fff; border: none; font-size: 12px;
  }
  .controls { flex: 1 1 260px; display: flex; flex-direction: column; gap: 12px; }
  select, button {
    font: inherit; padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--panel2); color: var(--text);
  }
  .btn-row { display: flex; gap: 10px; }
  button.primary { background: var(--accent); border: none; color: #04101f; font-weight: 600; cursor: pointer; flex: 1; }
  button.ghost { background: transparent; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .meta { color: var(--muted); font-size: 13px; margin-top: 2px; }
  #error-banner {
    display: none; background: rgba(255,93,93,.12); border: 1px solid rgba(255,93,93,.4);
    color: var(--fail); padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 16px;
  }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; }
  .badge.pass { background: rgba(46,204,113,.15); color: var(--pass); }
  .badge.fail { background: rgba(255,93,93,.15); color: var(--fail); }
  .score-bar { height: 8px; border-radius: 4px; background: var(--panel2); overflow: hidden; margin-top: 8px; }
  .score-bar > div { height: 100%; background: linear-gradient(90deg, var(--pass), var(--fail)); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px; margin-top: 16px; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    cursor: pointer; transition: transform .1s;
  }
  .card:hover { transform: translateY(-2px); }
  .card img { width: 100%; height: 120px; object-fit: cover; display: block; background: #000; }
  .card .body { padding: 10px; }
  .card .row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
  .card .name { color: var(--muted); font-size: 11px; margin-top: 4px; word-break: break-all; }
  #score-results.grid { grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
  #score-results .card img { height: 200px; }
  #score-results .card { cursor: default; }
  #score-results .card:hover { transform: none; }
  .heatmap-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }
  .heatmap-pair img { width: 100%; height: 160px; object-fit: cover; display: block; background: #000; }
  .heatmap-pair .img-caption {
    background: var(--panel); color: var(--muted); font-size: 10px; text-align: center; padding: 3px 0;
  }
  .advisory {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);
    font-size: 12px; line-height: 1.5;
  }
  .advisory-title { font-weight: 700; margin-bottom: 4px; }
  .advisory-field { margin-top: 4px; color: var(--muted); }
  .advisory-field b { color: var(--text, #eee); }
  .advisory-field ul { margin: 4px 0 0 18px; padding: 0; }
  .metrology-table { width: 100%; border-collapse: collapse; margin-top: 4px; font-size: 11px; }
  .metrology-table th, .metrology-table td { padding: 3px 6px; text-align: left; border-bottom: 1px solid var(--border); }
  .metrology-table th { color: var(--muted); font-weight: 600; }
  .drift-gauge { display: flex; align-items: center; gap: 8px; font-size: 12px; }
  .drift-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .drift-nominal .drift-dot { background: var(--pass); }
  .drift-nominal span:nth-child(2) { color: var(--pass); font-weight: 700; }
  .drift-alert .drift-dot { background: var(--fail); box-shadow: 0 0 6px var(--fail); }
  .drift-alert span:nth-child(2) { color: var(--fail); font-weight: 700; }
  .correct { outline: 2px solid rgba(46,204,113,.5); }
  .wrong { outline: 2px solid rgba(255,93,93,.6); }
  .filter-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 4px; }
  .filter-row select { padding: 8px 10px; font-size: 13px; }
  #gallery-summary { color: var(--muted); font-size: 13px; }
  #empty-state { color: var(--muted); font-size: 13px; padding: 20px 0; }
  #lightbox {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,.75);
    align-items: center; justify-content: center; z-index: 50; padding: 24px;
  }
  #lightbox .box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    max-width: 560px; width: 100%; overflow: hidden;
  }
  #lightbox img { width: 100%; max-height: 60vh; object-fit: contain; background: #000; display: block; }
  #lightbox .body { padding: 16px 20px; }
  #lightbox dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 13px; margin: 10px 0 0; }
  #lightbox dt { color: var(--muted); }
  #lightbox dd { margin: 0; word-break: break-all; }
  #lightbox .close { float: right; background: none; border: none; color: var(--muted); font-size: 18px; cursor: pointer; }
</style>
</head>
<body>
<header>
  <h1>AdaptiveVision -- PaDiM Anomaly Dashboard</h1>
  <p>Live scoring via the real OnnxInferenceEngine + ThresholdAnomalyDetector production code.</p>
  <div id="stat-chips"></div>
</header>
<main>
  <div id="error-banner"></div>

  <section>
    <h2>Test images</h2>
    <div class="panel">
      <div class="upload-row">
        <div id="dropzone">
          <div id="dz-hint">Drag &amp; drop one or more images, or click to choose</div>
          <div id="file-strip"></div>
          <input id="file-input" type="file" accept="image/*" multiple style="display:none">
        </div>
        <div class="controls">
          <label for="model-select" class="meta">Model</label>
          <div class="btn-row">
            <select id="model-select" style="flex:1"></select>
            <button class="ghost" id="reload-btn" title="Rescan models/ for newly trained models">&#8635;</button>
          </div>
          <div class="btn-row">
            <button class="primary" id="run-btn" disabled>Run inspection</button>
            <button class="ghost" id="clear-btn">Clear</button>
          </div>
        </div>
      </div>
      <div id="score-results" class="grid"></div>
    </div>
  </section>

  <section>
    <h2>Sample results gallery</h2>
    <div class="filter-row">
      <select id="filter-dataset"><option value="all">All datasets</option></select>
      <select id="filter-verdict">
        <option value="all">All verdicts</option>
        <option value="pass">Pass only</option>
        <option value="fail">Fail only</option>
      </select>
      <select id="filter-sort">
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="score_desc">Highest score first</option>
        <option value="score_asc">Lowest score first</option>
      </select>
      <span id="gallery-summary"></span>
    </div>
    <div id="empty-state" style="display:none">No results yet -- run <code>training/push_results_to_dashboard.py</code>.</div>
    <div id="gallery" class="grid"></div>
  </section>
</main>

<div id="lightbox">
  <div class="box">
    <img id="lb-img" alt="">
    <div class="body">
      <button class="close" id="lb-close">&times;</button>
      <span id="lb-badge"></span>
      <dl id="lb-meta"></dl>
    </div>
  </div>
</div>

<script>
function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function showError(msg) {
  const banner = document.getElementById('error-banner');
  banner.textContent = msg;
  banner.style.display = 'block';
  clearTimeout(showError._t);
  showError._t = setTimeout(() => { banner.style.display = 'none'; }, 6000);
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  let data = null;
  try { data = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) {
    const detail = (data && (data.detail || data.error)) || res.statusText || 'request failed';
    throw new Error(detail);
  }
  return data;
}

// ---------- Upload + scoring ----------
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileStrip = document.getElementById('file-strip');
const modelSelect = document.getElementById('model-select');
const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const reloadBtn = document.getElementById('reload-btn');
const scoreResults = document.getElementById('score-results');
let selectedFiles = [];
let allModels = [];

dropzone.addEventListener('click', e => { if (e.target === dropzone || e.target.id === 'dz-hint') fileInput.click(); });
['dragenter', 'dragover'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', e => addFiles(e.dataTransfer.files));
fileInput.addEventListener('change', e => addFiles(e.target.files));

function addFiles(fileList) {
  const images = Array.from(fileList || []).filter(f => f.type.startsWith('image/'));
  if (!images.length) { showError('Please choose image files.'); return; }
  selectedFiles = selectedFiles.concat(images);
  renderFileStrip();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderFileStrip();
}

function renderFileStrip() {
  fileStrip.innerHTML = '';
  document.getElementById('dz-hint').style.display = selectedFiles.length ? 'none' : 'block';
  selectedFiles.forEach((file, i) => {
    const div = document.createElement('div');
    div.className = 'thumb';
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    const btn = document.createElement('button');
    btn.textContent = '\u00D7';
    btn.title = 'Remove';
    btn.onclick = ev => { ev.stopPropagation(); removeFile(i); };
    div.appendChild(img);
    div.appendChild(btn);
    fileStrip.appendChild(div);
  });
  runBtn.disabled = selectedFiles.length === 0 || allModels.length === 0;
}

clearBtn.addEventListener('click', () => {
  selectedFiles = [];
  scoreResults.innerHTML = '';
  renderFileStrip();
});

reloadBtn.addEventListener('click', async () => {
  reloadBtn.disabled = true;
  try {
    await fetchJson('/api/reload', { method: 'POST' });
    await loadModels();
  } catch (err) {
    showError('Reload failed: ' + err.message);
  } finally {
    reloadBtn.disabled = false;
  }
});

function chipClass(auroc) {
  if (auroc == null) return '';
  if (auroc >= 0.9) return 'good';
  if (auroc >= 0.7) return 'mid';
  return 'bad';
}

async function loadModels() {
  allModels = await fetchJson('/api/models');
  const chips = document.getElementById('stat-chips');
  const datasetFilter = document.getElementById('filter-dataset');

  if (!allModels.length) {
    modelSelect.innerHTML = '<option value="">No trained models found</option>';
    chips.innerHTML = '<span class="chip">No models -- run training/train_padim.py</span>';
    runBtn.disabled = true;
    return;
  }

  modelSelect.innerHTML = allModels.map(m => {
    const auroc = m.test_auroc != null ? (m.test_auroc * 100).toFixed(1) + '%' : 'n/a';
    const label = `${m.dataset}${m.category ? '/' + m.category : ''} · ${m.backbone} · AUROC ${auroc}`;
    return `<option value="${escapeHtml(m.name)}">${escapeHtml(label)}</option>`;
  }).join('');

  chips.innerHTML = allModels.map(m => {
    const auroc = m.test_auroc != null ? (m.test_auroc * 100).toFixed(1) + '%' : 'n/a';
    const label = `${m.dataset}${m.category ? '/' + m.category : ''}`;
    return `<span class="chip ${chipClass(m.test_auroc)}"><b>${escapeHtml(label)}</b> ${auroc}</span>`;
  }).join('');

  const datasets = [...new Set(allModels.map(m => m.dataset))].sort();
  datasetFilter.innerHTML = '<option value="all">All datasets</option>' +
    datasets.map(d => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join('');

  runBtn.disabled = selectedFiles.length === 0;
}

runBtn.addEventListener('click', async () => {
  if (!selectedFiles.length || !modelSelect.value) return;
  runBtn.disabled = true;
  runBtn.textContent = 'Scoring...';
  const form = new FormData();
  selectedFiles.forEach(f => form.append('files', f));
  form.append('model', modelSelect.value);
  try {
    const data = await fetchJson('/api/score', { method: 'POST', body: form });
    renderScoreResults(data);
  } catch (err) {
    showError('Scoring failed: ' + err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run inspection';
  }
});

function renderScoreResults(data) {
  scoreResults.innerHTML = data.results.map(r => {
    if (r.error) {
      return `<div class="card"><div class="body">
        <div class="meta">${escapeHtml(r.filename)}</div>
        <div class="meta" style="color:var(--fail)">${escapeHtml(r.error)}</div>
      </div></div>`;
    }
    const pct = Math.round(r.score * 100);
    const adv = r.advisory;
    const advisoryHtml = adv ? `
        <div class="advisory">
          <div class="advisory-title">🤖 AI explanation (${adv.is_fallback ? 'offline fallback' : 'llama3'}, confidence ${Math.round(adv.confidence_score * 100)}%)</div>
          <div class="advisory-field"><b>Classification:</b> ${escapeHtml(adv.defect_classification)}</div>
          <div class="advisory-field"><b>Likely cause:</b> ${escapeHtml(adv.root_cause_hypothesis)}</div>
          ${adv.recommended_actions && adv.recommended_actions.length ? `
          <div class="advisory-field"><b>Recommended actions:</b>
            <ul>${adv.recommended_actions.map(a => `<li>${escapeHtml(a)}</li>`).join('')}</ul>
          </div>` : ''}
        </div>` : '';
    const matches = r.similar_historical_defects || [];
    const retrievalHtml = matches.length ? `
        <div class="advisory">
          <div class="advisory-title">🔎 Similar historical defects</div>
          ${matches.map(m => `<div class="advisory-field">${escapeHtml(m.defect_type || 'unknown')} — ${escapeHtml(m.dataset || '')}/${escapeHtml(m.category || '')} (similarity ${Math.round((m.similarity || 0) * 100)}%)</div>`).join('')}
        </div>` : '';
    const heatmapHtml = r.heatmap ? `
        <div class="heatmap-pair">
          <div><img src="${r.thumbnail}" alt="original"><div class="img-caption">original</div></div>
          <div><img src="${r.heatmap}" alt="heatmap"><div class="img-caption">strongest signal: ${escapeHtml(r.heatmap_region || 'unknown')} (approximate)</div></div>
        </div>` : `<img src="${r.thumbnail}" alt="">`;
    const measurements = r.defect_measurements || [];
    const metrologyHtml = measurements.length ? `
        <div class="advisory">
          <div class="advisory-title">📏 Defect metrology</div>
          <table class="metrology-table">
            <thead><tr><th>#</th><th>type</th><th>bbox (x,y,w,h)</th><th>area</th><th>aspect</th></tr></thead>
            <tbody>
              ${measurements.map((m, i) => `<tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(m.morphology)}</td>
                <td>${m.bbox.join(', ')}</td>
                <td>${m.area_um2.toFixed(1)} µm²</td>
                <td>${m.aspect_ratio.toFixed(2)}:1</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>` : '';
    const drift = r.drift_report;
    const driftAlert = r.drift_status === 'SENSOR_DRIFT_ALERT';
    const driftHtml = drift ? `
        <div class="advisory">
          <div class="advisory-title">📡 Optical drift monitor</div>
          <div class="drift-gauge ${driftAlert ? 'drift-alert' : 'drift-nominal'}">
            <span class="drift-dot"></span>
            <span>${driftAlert ? 'SENSOR DRIFT ALERT' : 'NOMINAL'}</span>
            <span class="meta">(p=${drift.p_value.toFixed(4)}, window ${drift.window_size}/${drift.baseline_size})</span>
          </div>
        </div>` : '';
    return `<div class="card">
      ${heatmapHtml}
      <div class="body">
        <div class="row">
          <span class="badge ${r.is_anomalous ? 'fail' : 'pass'}">${r.verdict}</span>
          <span class="meta">${pct}%</span>
        </div>
        <div class="score-bar"><div style="width:${pct}%"></div></div>
        <div class="meta">resized to ${r.resized_to}</div>
        <div class="name">${escapeHtml(r.filename)}</div>
        ${advisoryHtml}
        ${retrievalHtml}
        ${metrologyHtml}
        ${driftHtml}
      </div>
    </div>`;
  }).join('');
}

// ---------- Gallery ----------
const galleryEl = document.getElementById('gallery');
const summaryEl = document.getElementById('gallery-summary');
const emptyState = document.getElementById('empty-state');
let galleryItems = [];

async function loadGallery() {
  const dataset = document.getElementById('filter-dataset').value;
  const verdict = document.getElementById('filter-verdict').value;
  const sort = document.getElementById('filter-sort').value;
  const params = new URLSearchParams({ limit: 300, dataset, verdict, sort });
  let totalMatching = 0;
  try {
    const data = await fetchJson('/api/gallery?' + params.toString());
    galleryItems = data.items;
    totalMatching = data.total_matching;
  } catch (err) {
    showError('Failed to load gallery: ' + err.message);
    return;
  }

  if (!galleryItems.length) {
    galleryEl.innerHTML = '';
    emptyState.style.display = 'block';
    summaryEl.textContent = '';
    return;
  }
  emptyState.style.display = 'none';

  let correct = 0, labeled = 0;
  galleryEl.innerHTML = galleryItems.map((item, idx) => {
    const predictedAnomaly = item.verdict === 'fail';
    const gtAnomaly = item.ground_truth === 'anomaly';
    let correctness = '';
    if (item.ground_truth) {
      labeled++;
      const isCorrect = predictedAnomaly === gtAnomaly;
      if (isCorrect) correct++;
      correctness = isCorrect ? 'correct' : 'wrong';
    }
    const img = item.thumbnail || '';
    return `<div class="card ${correctness}" data-idx="${idx}">
      <img src="${img}" alt="" loading="lazy">
      <div class="body">
        <div class="row">
          <span class="badge ${predictedAnomaly ? 'fail' : 'pass'}">${item.verdict.toUpperCase()}</span>
          <span class="meta">${Math.round(item.anomaly_score * 100)}%</span>
        </div>
        <div class="meta">gt: ${escapeHtml(item.ground_truth || 'n/a')} · ${escapeHtml(item.dataset)}</div>
        <div class="name">${escapeHtml(item.source_name || '')}</div>
      </div>
    </div>`;
  }).join('');

  const shownLabel = totalMatching > galleryItems.length
    ? `${galleryItems.length} of ${totalMatching} shown`
    : `${galleryItems.length} shown`;
  summaryEl.textContent = labeled
    ? `${shownLabel} · ${correct}/${labeled} correct (${Math.round(100 * correct / labeled)}%)`
    : shownLabel;

  galleryEl.querySelectorAll('.card').forEach(card => {
    card.addEventListener('click', () => openLightbox(galleryItems[Number(card.dataset.idx)]));
  });
}

['filter-dataset', 'filter-verdict', 'filter-sort'].forEach(id =>
  document.getElementById(id).addEventListener('change', loadGallery));

// ---------- Lightbox ----------
const lightbox = document.getElementById('lightbox');
function openLightbox(item) {
  document.getElementById('lb-img').src = item.thumbnail || '';
  document.getElementById('lb-badge').innerHTML =
    `<span class="badge ${item.verdict === 'fail' ? 'fail' : 'pass'}">${item.verdict.toUpperCase()}</span>`;
  document.getElementById('lb-meta').innerHTML = `
    <dt>Score</dt><dd>${(item.anomaly_score * 100).toFixed(1)}%</dd>
    <dt>Ground truth</dt><dd>${escapeHtml(item.ground_truth || 'n/a')}</dd>
    <dt>Dataset</dt><dd>${escapeHtml(item.dataset)}</dd>
    <dt>Model</dt><dd>${escapeHtml(item.model_ver)}</dd>
    <dt>File</dt><dd>${escapeHtml(item.source_name || '')}</dd>
    <dt>Inspected</dt><dd>${escapeHtml(item.timestamp_utc)}</dd>
  `;
  lightbox.style.display = 'flex';
}
function closeLightbox() { lightbox.style.display = 'none'; }
document.getElementById('lb-close').addEventListener('click', closeLightbox);
lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ---------- Boot ----------
(async function init() {
  try {
    await loadModels();
  } catch (err) {
    showError('Failed to load models: ' + err.message);
  }
  await loadGallery();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
