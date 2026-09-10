# Architecture

Two things share one library: a runtime inspection application (`src/adaptivevision/`) and an
offline research pipeline (`training/`). They meet at the ONNX model files.

## 1. Working runtime path

This is what `python scripts/run_station.py` and the teach-mode upload actually execute. Every stage
is a callable injected by the composition root (`src/adaptivevision/app.py::build_station`); an
unconfigured stage is skipped.

```
image  (camera.py: NullCameraDriver / FileImageCameraDriver / InMemoryCameraDriver)
  -> RawFrame
preprocess  (camera.py: ensure_grayscale?, resize_to(H, W))          [MODEL_INPUT_HEIGHT/WIDTH]
  -> RawFrame
rectify  (camera.py: CalibrationRectifier.apply)                     [optional; identity copy today]
  -> RectifiedFrame
align  (camera.py: ReferenceAligner.align)                           [optional; returns a fixed pose today]
  -> LocalizedPart
anomaly detect  (metrology.py: ThresholdAnomalyDetector.detect)
     - float32 cast, (H,W)->(1,H,W) or (H,W,C)->(C,H,W)
     - engine.py: OnnxInferenceEngine.infer({"input": ...}) -> "output" scalar in [0,1]
     - score >= recipe threshold  ->  Defect(ANOMALY, severity)
     - if a NormalPatchBank + MetrologyConfig are configured and the frame is anomalous:
         NormalPatchBank.heatmap(...) -> measure_defects(...) -> DefectMeasurement[]
  -> AnomalyResult(score, threshold, is_anomalous, defects, defect_measurements, heatmap_region, heatmap)
decide  (decision.py: DecisionPolicy.decide)
     - any defect at or above fail_severity            -> FAIL
     - more defects than max_defects                   -> FAIL
     - any remaining defect                            -> REVIEW
     - non-anomalous but score in a review band        -> REVIEW
     - otherwise                                       -> PASS
  -> Decision(verdict, defects)
build result  (orchestration.py: InspectionPipeline.run)
  -> InspectionResult   (verdict, defects, anomaly_score, lineage, cycle_time_ms, image_refs, ...)
```

After the cycle clock stops, the result goes to an `on_result` chain (`app.py::chain_handlers`),
none of which can stall or fail the inspection:

```
InspectionResult
  -> storage.py: PersistenceHandler.on_result  -> SqliteResultRepository.save_result  -> adaptivevision.db
       and (if configured) LocalImageStore.archive_image -> images/<frame_id>.bin  (+ -heatmap.bin)
  -> communication.py: RejectDispatcher.on_result  -> ModbusTcpTransport.write_coil   [FAIL only; logs and continues if no PLC]
  -> explanation.py: AdvisoryWorker.on_result  -> background queue                    [FAIL / REVIEW only]
       -> OllamaAdvisoryEngine.generate_report -> Pydantic validate -> retry -> deterministic fallback
       -> SqliteAdvisoryRepository.save_report -> adaptivevision.db
```

Serving (`api.py::create_app`, run by `scripts/run_api.py` or `run_dashboard.py`):

```
adaptivevision.db  ->  GET /api/v1/results , GET /api/v1/results/{id}
images/            ->  GET /api/v1/images/{id}
advisory_reports   ->  GET /api/v1/advisory/{id}
GET /              ->  embedded HTML dashboard (polls the JSON routes)
POST /api/v1/inspect-upload  ->  ManualInspectionService.inspect  (same pipeline as above, in a threadpool)
GET /metrics       ->  Prometheus text (the registry is not populated by the running station -- see below)
GET /ws/results    ->  WebSocket; accepts connections, no server-side broadcast is wired
```

### The five modules to read first

| Module | Responsibility |
|---|---|
| `src/adaptivevision/orchestration.py` | one inspection cycle (`InspectionPipeline.run`) and the station state machine |
| `src/adaptivevision/app.py` | the composition root: turns config into a wired `StationController` |
| `src/adaptivevision/engine.py` | the ONNX Runtime adapter (`OnnxInferenceEngine`) and latency measurement |
| `src/adaptivevision/metrology.py` | the anomaly detector, Otsu + connected-components defect metrology, the patch bank |
| `src/adaptivevision/decision.py` | the deterministic PASS / FAIL / REVIEW rules |

## 2. Offline research path

`training/` never imports `src/adaptivevision` except `engine.py` (to verify an export) and a few
value objects. It needs `torch`, `timm`, and (for some steps) `scikit-learn`, `scipy`, `pandas`.

```
datasets (MVTec AD / VisA / KolektorSDD2 / Severstal)   -- training/data.py: discover + load paths, labels, masks
  -> FrozenFeatureExtractor (timm backbone)             -- training/models.py
  -> a Scorer: PatchCoreScorer | PaDiMScorer | DFMScorer -- training/models.py  (training-free fit)
                or DinomalyScorer                        -- training/models.py  (gradient training, GPU)
  -> evaluate: compute_metrics / compute_pixel_metrics   -- training/evaluate.py  (AUROC, AP, F1, AUPRO, AUPIMO, escape/overkill)
  -> training/store.py: one row per (method, regime, config, seed) in a SQLite results DB
                        + prediction .npz artifacts + model checkpoints
  -> training/sweep.py drives the whole grid (one-class / multi-class / few-shot regimes), resumable
  -> training/evaluate.py leaderboard: aggregate the DB into ranking + winners CSVs + leaderboard.md
  -> training/export.py:
        export           fit -> calibrate on held-out normals -> torch.onnx.export -> verify through OnnxInferenceEngine
        quantize         dynamic weights-only INT8 (+ memory-bank rebuild for PatchCore)
        retrieval-index  embed anomalous test images -> FaissRetrievalIndex.save
        deployment-export sweep DB -> deployment_profiles.json  (read by src/adaptivevision/deployment.py)
  -> models/*.onnx  (+ models/*.json manifests)  <-- consumed by the runtime path above
```

`training/report.py` builds a single static HTML report from a finished sweep.
`training/dashboard_app.py` is a separate FastAPI app for visually inspecting results and uploading
images; it is the only place that chains FAISS retrieval + the LLM advisory into one request.

## 3. Experimental or partially connected

Implemented and unit-tested, but **not** part of the running inspection pipeline:

| Piece | Where | State |
|---|---|---|
| Drift detection (two-sample KS test) | `drift.py::DriftDetector`, `ks_two_sample` | used only by `training/dashboard_app.py` |
| Metrics registry + Prometheus exposition | `drift.py::MetricsRegistry`, `render_metrics`, `GET /metrics` | endpoint responds; nothing in the station calls `increment()` / `set_gauge()` |
| SPC control chart | `drift.py::control_chart` | not called outside tests |
| Component health checks | `drift.py::HealthCheck`, `GET /api/v1/health` | endpoint responds; no probes registered |
| Calibration self-test + hot-swap | `camera.py::CalibrationSelfTest`, `CalibrationManager` | not wired into `build_station` |
| Threaded frame buffer | `camera.py::ThreadedFrameBuffer` | not constructed anywhere |
| Failure-retry buffer | `orchestration.py::ResultBuffer`, `FailureHandler` | not used; `PersistenceHandler` does its own log-and-swallow |
| Cycle watchdog | `orchestration.py::CycleWatchdog` | built and injected, but `.check()` is never called |
| MQTT publisher | `communication.py::MqttPublisher` | no build function; never constructed |
| Dimensional metrology inspector | `metrology.py::MetrologyInspector` | no measurement source; not wired |
| FAISS retrieval | `explanation.py::FaissRetrievalIndex` | only in `training/dashboard_app.py` |
| Deployment recommender | `deployment.py`, `GET /api/v1/deployment/recommendation` | code + tests fine; needs `deployment_profiles.json`, which is generated by `training/export.py deployment-export` |
| Optical rectification / alignment | `camera.py::CalibrationRectifier`, `ReferenceAligner` | placeholders: identity image copy, fixed nominal pose |
| Docker / Prometheus / Grafana | `deploy/` | configuration only, not built or run |

Keeping these here (rather than deleting them) is a deliberate choice: they show the intended shape
of a full station, and each is a small, tested unit. The list above is the honest map of what is and
is not on the live path.
