# Architecture (pointer document)

This file is the in-repo pointer to the **frozen** design baseline.

- **Source of truth:** *AdaptiveVision — Architecture Specification v1.0*.
- **Build plan:** *AdaptiveVision — Implementation Roadmap v1.0*.

Both documents are frozen. Nothing in this repository renames or restructures
what they define. Any change that genuinely requires altering the architecture
must go through a change request and bump the spec to v1.1 before code changes.

## Layering (summary)

The system is organized into layers with dependencies pointing inward toward the
domain and its interfaces (`common/interfaces.py`):

1. **Presentation** — dashboard web app.
2. **Interface** — FastAPI + WebSocket (`api/`).
3. **Orchestration** — station controller, state machine, pipeline, scheduler,
   watchdog (`orchestration/`, `station.py`, `app.py`).
4. **Domain** — acquisition, calibration, preprocessing, alignment, inference,
   inspection, decision, recipe.
5. **Infrastructure** — communication (Modbus/MQTT), persistence, monitoring,
   internal event bus.
6. **Cross-cutting** — configuration, logging, shared types/errors/interfaces.

## Core design principle

The inspection loop is **deterministic and isolated**; I/O, persistence, and
dashboard updates are **asynchronous and off the critical path**. A slow broker
or a full disk must never stall a part inspection.

## Milestone records

Per-milestone implementation notes are below.

---

## Milestones

Per-milestone implementation notes, M0 through M20 (M2's scope was
folded into M1 and M3; there is no separate M2 entry).

### Contents

- Milestone M0 — Project Scaffolding & Tooling Foundation
- Milestone M1 - Domain Contracts & Interfaces
- Milestone M3 — Walking Skeleton
- Milestone M4 - Persistence & Traceability
- Milestone M5 - Preprocessing and Calibration
- Milestone M6 - Alignment
- Milestone M7 - Metrology Inspection
- Milestone M8 - ONNX Inference Engine
- Milestone M9 - Anomaly Inspection
- Milestone M10 - Decision Fusion & Policy
- Milestone M11 - PLC / Modbus
- Milestone M12 - MQTT
- Milestone M13 - Minimal API + Dashboard
- Milestone M14 - Monitoring / Metrics / SPC / Health
- Milestone M15 - Full API + Dashboard
- Milestone M16 - Calibration Lifecycle + Hot-Swap + Self-Test
- Milestone M17 - Failure Handling + Buffering + Regression/Performance Suites
- Milestone M18 - Deployment / Containers / Edge
- Milestone M19 - Quality Intelligence & Deployment Observability
- Milestone M20 - Wire the M9/M10 Inspection Chain Into the Live Station

### Milestone M0 — Project Scaffolding & Tooling Foundation

**Goal.** A repository that installs, lints, type-checks, runs an (initially
small) test suite, and boots to a single logged, structured line.

#### Delivered

- Frozen package tree from Architecture Spec v1.0 §7.1 (packages +
  docstring-only `__init__.py`; no future-module code).
- `logging_setup.py`: structured JSON logging with a context-scoped
  `correlation_id` and an idempotent `configure_logging()`.
- `scripts/run_station.py`: minimal boot entrypoint that logs one structured
  line and exits `0`.
- Tooling gate: `pyproject.toml` (hatchling packaging; ruff, mypy-strict,
  pytest+coverage config), pre-commit hooks, GitHub Actions CI, `Makefile`.
- `.env.example`, `.gitignore`, `README.md`, `docs/` skeleton.
- Tests: `logging_setup` unit tests, package smoke test, and an e2e test of the
  boot entrypoint.

#### Explicitly out of scope (deferred to later milestones)

- Any domain logic, interfaces, or types (`common/*` → M1).
- Configuration loading and recipe schema (`config/*`, `recipe/*` → M2).
- The composition root and state machine (`app.py`, `station.py`,
  `orchestration/*` → M3).

#### Acceptance criteria (met)

Fresh clone → `pip install -e ".[dev]"` → `ruff check .` and `ruff format
--check .` clean → `mypy` clean → `pytest` green → `python
scripts/run_station.py` emits one JSON line containing a populated
`correlation_id` field and exits `0`.

---

### Milestone M1 - Domain Contracts & Interfaces

**Goal.** Freeze the shared domain vocabulary (value types, enums, error
taxonomy, IDs, pure geometry, timing) and the abstraction seams every later
milestone builds against. Contracts, not behaviour.

#### Delivered (`src/adaptivevision/common/`)

- `enums.py` - closed, string-valued enumerations (`Verdict`, `StationState`,
  `Severity`, `DefectClass`, `CameraKind`, `ExecutionProvider`). Values pinned.
- `errors.py` - exception hierarchy under `AdaptiveVisionError` with a
  `recoverable` / `is_fatal` contract (Spec v1.0 §17).
- `ids.py` - time-ordered identifier generation, independent of logging.
- `geometry.py` - pure-Python 2D geometry (no NumPy/OpenCV).
- `timing.py` - `Stopwatch` / `Deadline` / `measure` on `time.monotonic`,
  injectable clock.
- `types.py` - frozen value objects: `ROI`, `Pose`, `Tolerance`,
  `MeasurementSpec`, `Measurement`, `RawFrame`, `RectifiedFrame`.
- `result.py` - `Defect`, `PartialResult` base, `MetrologyResult`,
  `AnomalyResult`, `ClassicalResult`, `InspectionResult` (lossless round-trip).
- `interfaces.py` - the eight ABCs (`CameraDriver`, `InferenceEngine`,
  `AnomalyDetector`, `Inspector`, `PLCTransport`, `MessagePublisher`,
  `ResultRepository`, `RecipeStore`). `Inspector` and `RecipeStore` are generic
  over the M6/M2 aggregates so no future types are invented.
- `common/__init__.py` re-exports the public surface.

#### Frozen implementation decisions honored

ABCs (not Protocols); explicit string enums; pure-Python geometry; frozen
dataclasses; explicit `to_dict`/`from_dict`; `Tolerance`/`MeasurementSpec` in
`types.py`; `time.monotonic` in `timing.py`; `ids.py` independent of logging;
`common/` free of runtime NumPy (image type resolved only under
`TYPE_CHECKING`).

#### Explicitly out of scope (later milestones)

Config/recipe schema (M2), acquisition/orchestration (M3), and all concrete
implementations of the interfaces. No `LocalizedPart` (M6) or `Recipe` (M2)
types were created.

#### Notes for M2

- The recipe (M2) composes on `ROI`, `MeasurementSpec`/`Tolerance`, the decision
  enums, and raises `RecipeError`; `RecipeStore[Recipe]` binds the generic.
- Inspector references (M2 recipe) should be validated strings against a
  registry - no new enum was introduced (avoids a spec change).
- `ResultRepository` storage failures currently surface as the base
  `AdaptiveVisionError`; a dedicated `PersistenceError` is a candidate change
  request at M4.

#### Acceptance criteria (met)

`geometry.py` fully covered with numeric assertions; `InspectionResult`
round-trips losslessly; all eight interfaces import cleanly, are abstract, and
are documented; ruff / black / mypy(strict) clean; pytest green at 100% for M1
modules.

---

### Milestone M3 — Walking Skeleton

**Goal.** Boot the station end-to-end: the composition root wires a null-object
camera driver into the orchestration layer (state machine, inspection pipeline,
scheduler, cycle watchdog), and `scripts/run_station.py` runs a demo inspection
cycle against the synthetic camera, then shuts down cleanly.

#### Delivered

- `adaptivevision/app/` — the composition root (`app.py`) and the station
  controller (`station.py`).
  - `build_station(config)` is the only place that wires concrete
    implementations to the abstraction seams (Spec v1.0 §19). It reads the
    validated `StationConfig`, builds the camera driver, and assembles the
    orchestration layer into a `StationController`.
  - `build_camera(config)` uses the null-object strategy: when no camera is
    configured, a synthetic 640x480 `NullCameraDriver` is returned so the
    walking skeleton runs without hardware.
  - `StationController` owns the state machine and drives the pipeline through
    the scheduler, enforcing cycle-time limits with the watchdog. It exposes the
    stable `boot() / ready() / run() / shutdown()` lifecycle.
- `adaptivevision/acquisition/` — `NullCameraDriver`, the synthetic
  `CameraDriver` implementation that is always healthy, opens/closes without
  side effects, and produces a zero-filled grayscale frame of the configured
  size. Real camera backends replace it behind the same seam in later
  milestones.
- `adaptivevision/orchestration/` — the orchestration layer:
  - `state.py` — `StationStateMachine`, a guarded FSM over `StationState`
    modelling the boot path (`INIT -> SELF_TEST -> IDLE -> READY -> RUNNING`)
    and the fault / shutdown paths. Invalid transitions raise `FaultError`.
  - `pipeline.py` — `InspectionPipeline`, the heart of the walking skeleton: it
    drives one inspection cycle by acquiring a frame from the camera driver and
    producing an `InspectionResult`.
  - `scheduler.py` — `InspectionScheduler`, a simple synchronous driver that
    runs a bounded number of cycles against the pipeline and reports each
    result.
  - `watchdog.py` — `CycleWatchdog`, which enforces the maximum allowed cycle
    time and counts timeout violations.
- `scripts/run_station.py` — the boot entrypoint: loads and validates
  configuration, configures structured logging, builds the station through the
  composition root, boots it through the state machine, runs a short demo
  inspection cycle against the null-object camera, and shuts down cleanly.
- Tests:
  - `tests/unit/test_acquisition.py` — null-object camera driver behaviour.
  - `tests/unit/test_orchestration.py` — state machine, pipeline, scheduler, and
    watchdog.
  - `tests/e2e/test_run_station.py` — boots the entrypoint and asserts the
    structured boot / inspection / shutdown log lines.

#### Explicitly out of scope (deferred to later milestones)

- Configuration loading and recipe schema (`config/*`, `recipe/*` → M2).
- Local persistence layer (`persistence/*` → M4).
- Preprocessing and calibration rectification wiring (`preprocessing/*`,
  `calibration/*` → M5).
- Golden-reference alignment (`alignment/*` → M6).
- Metrology inspection (`inspection/metrology/*` → M7).
- AI inference (`inference/*` → M8).
- Anomaly detection (`inspection/anomaly/*` → M9).
- Decision fusion and policy (`decision/*` → M10).
- PLC / MQTT integration (`communication/*` → M11/M12).
- HTTP API and dashboard (`api/*`, `dashboard/*` → M13).
- Monitoring / metrics / SPC / health (`monitoring/*` → M14).
- Calibration lifecycle and hot-swap (`calibration/lifecycle.py` → M16).
- Failure handling and buffering (`orchestration/failure.py`,
  `orchestration/buffer.py` → M17).
- Deployment / containers / edge (`deploy/*` → M18).

#### Acceptance criteria (met)

`python scripts/run_station.py` boots the walking skeleton, runs a demo
inspection cycle against the null-object camera, and shuts down cleanly, emitting
structured JSON log lines with a populated `correlation_id` and milestone `M3`.
The e2e test asserts the boot / inspection / shutdown lines and exit code `0`.
ruff / black / mypy (strict) are clean and pytest is green for the M3 modules.

---

### Milestone M4 - Persistence & Traceability

**Goal.** Persist inspection results to a local edge database while preserving
traceability lineage and keeping persistence failures off the inspection
critical path.

#### Delivered

- SQLite database setup with SQLAlchemy engine/session helpers and idempotent
  schema initialization.
- ORM persistence model for inspection results with first-class lineage fields
  and JSON payloads for measurements, defects, image references, and traceability.
- `SqliteResultRepository` implementing result save/get/list behavior at the
  persistence boundary.
- Traceability record serialization for inspection, part, station, recipe,
  model, calibration, verdict, timing, defects, measurements, anomaly score, and
  image references.
- Bounded local image archive for M4 trace image references.
- Persistence handler suitable for the scheduler `on_result` callback; failures
  are logged and swallowed so the inspection loop continues safely.
- Integration tests covering the database layer, repository round trips,
  traceability, image archive, and handler failure behavior.

#### Explicitly out of scope

- Calibration/preprocessing image transforms (M5).
- Alignment and localized part state (M6).
- Metrology/anomaly/decision logic (M7-M10).
- Advanced failure buffering, WAL/retry queues, and performance suites (M17).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M5 - Preprocessing and Calibration

**Goal.** Load versioned camera calibration artifacts, apply calibration lineage
to acquired frames, and provide deterministic preprocessing operators before
later alignment and inspection stages.

#### Delivered

- Immutable `CameraCalibration` artifact model with explicit validation and
  JSON round-trip serialization.
- Calibration JSON loader that raises `CalibrationError` on missing, malformed,
  or invalid artifacts.
- `CalibrationRectifier` that validates camera/dimension compatibility and
  returns `RectifiedFrame` objects carrying the applied calibration version.
- Deterministic preprocessing operators for grayscale conversion and uint8
  normalization, plus an ordered preprocessing pipeline.
- Optional composition-root wiring through `CALIBRATION_PATH` and injected
  preprocessing/rectification callables.
- Pipeline lineage update: calibrated runs record `calib_ver` in the
  `InspectionResult`; uncalibrated skeleton runs remain supported.

#### Explicitly out of scope

- Alignment/localized part state (M6).
- Metrology inspection and tolerance evaluation (M7).
- Optical map generation, calibration lifecycle, hot-swap, and self-test (M16).
- Failure buffering/performance suites (M17).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M6 - Alignment

**Goal.** Localize rectified frames against versioned golden references and
produce an aligned-part aggregate for downstream inspection stages.

#### Delivered

- Immutable `GoldenReference` artifact model with validation and JSON
  serialization.
- `LocalizedPart` aggregate carrying the rectified frame, pose, reference
  lineage, and alignment score.
- Golden-reference JSON loader with explicit `FaultError` failures for missing,
  malformed, or invalid artifacts.
- `ReferenceAligner` that validates camera and image dimensions and emits
  deterministic alignment lineage.
- Optional composition-root wiring through `REFERENCE_PATH` and injected
  aligner callables.
- Pipeline alignment stage after preprocessing and rectification, without
  pulling M7 metrology or decision behavior forward.

#### Explicitly out of scope

- Metrology inspection and tolerance evaluation (M7).
- AI inference/anomaly detection (M8-M9).
- Decision fusion/policy application (M10).
- Calibration lifecycle hot-swap and self-test (M16).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M7 - Metrology Inspection

**Goal.** Evaluate dimensional measurement specifications against aligned
parts and produce metrology measurements and defects in real-world units.

#### Delivered

- `MetrologyInspector` implementing the existing `Inspector[LocalizedPart,
  Recipe]` seam.
- Injectable measurement source contract for deterministic replay, tests, and
  future real metrology tools.
- `StaticMeasurementSource` for simulated/replay measurements.
- Measurement evaluation against `MeasurementSpec.contains()` with recorded
  `Measurement.in_tolerance` outcomes.
- Dimensional defects for out-of-tolerance or missing measurements, avoiding
  silent PASS behavior.
- Optional pipeline integration that includes metrology measurements/defects in
  `InspectionResult` when an aligned part, recipe, and inspector are injected.

#### Explicitly out of scope

- ONNX inference (M8).
- Anomaly inspection (M9).
- Multi-signal decision fusion and recipe decision policy (M10).
- OCR/barcode/presence inspection and other V2 classical tools.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M8 - ONNX Inference Engine

**Goal.** Provide a production-shaped ONNX inference engine behind the existing
`InferenceEngine` abstraction.

#### Delivered

- `OnnxInferenceEngine` implementing model load, warmup, infer, unload, and
  `model_version` lineage.
- Lazy ONNX Runtime import so missing runtime dependencies raise explicit
  `InferenceError` instead of import-time crashes.
- Execution-provider mapping from the frozen `ExecutionProvider` enum to ONNX
  Runtime provider names.
- Model path resolution from a configured model directory.
- Output-name mapping from ONNX Runtime outputs to returned tensors.
- Unit tests using an injected ONNX Runtime-like fake so M8 verifies
  deterministically without downloading external dependencies.

#### Explicitly out of scope

- Anomaly inspection policy and heatmaps (M9).
- TensorRT/INT8 optimization and provider-specific tuning (V2 backlog).
- Active learning, classifier, segmentation, and model registry workflows.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M9 - Anomaly Inspection

**Goal.** Score rectified frames for anomalies and produce anomaly results with
a decision threshold, heatmap reference, and anomaly defects.

#### Delivered

- `StaticAnomalyDetector` implementing the existing `AnomalyDetector` seam for
  deterministic replay, tests, and simulated stations.
- `ThresholdAnomalyDetector` that wraps the M8 `InferenceEngine`, batches a
  2D frame image, extracts a scalar anomaly score, and applies a decision
  threshold.
- Anomaly defects (`DefectClass.ANOMALY`) raised when the score meets or
  exceeds the threshold, avoiding silent PASS behavior.
- Optional heatmap reference carried on the `AnomalyResult`.
- Optional pipeline integration that records `anomaly_score` and anomaly
  defects in `InspectionResult` when an anomaly detector is injected.

#### Explicitly out of scope

- Multi-signal decision fusion and recipe decision policy (M10).
- PLC / Modbus transport (M11).
- MQTT messaging (M12).
- Supervised defect classification and classical AOI tools (V2 backlog).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M10 - Decision Fusion & Policy

**Goal.** Fuse the partial results produced by the inspection inspectors into a
single deterministic verdict.

#### Delivered

- `DecisionPolicy` that fuses partial results (metrology M7, anomaly M9,
  classical AOI) into a `Verdict`.
- `Decision` value object carrying the fused verdict and all fused defects.
- Severity-based policy:
  - Any CRITICAL or MAJOR defect -> FAIL.
  - Otherwise any MINOR or INFO defect -> REVIEW.
  - Otherwise, an anomaly score in a configured review band -> REVIEW.
  - Otherwise -> PASS.
- Optional anomaly review threshold for routing near-threshold parts to REVIEW.
- Pipeline integration: the pipeline uses the injected `DecisionPolicy` to
  compute the verdict from the metrology and anomaly partial results, falling
  back to the legacy FAIL-if-any-defect rule when no policy is provided.

#### Explicitly out of scope

- PLC / Modbus transport (M11).
- MQTT messaging (M12).
- Supervised defect classification and classical AOI tools (V2 backlog).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M11 - PLC / Modbus

**Goal.** Provide a Modbus TCP transport for PLC register and coil access.

#### Delivered

- `ModbusTcpTransport` implementing the existing `PLCTransport` seam.
- Pluggable `ModbusClient` protocol for the low-level wire protocol, keeping
  the transport free of external dependencies and testable with a fake client.
- Connection lifecycle (`connect` / `disconnect` / `is_connected`).
- Coil and holding-register read/write operations.
- All client failures translated to `CommsError`; operations require an
  established connection.

#### Explicitly out of scope

- MQTT messaging (M12).
- Station-level PLC handshake / recipe selection logic (orchestration).
- Supervised defect classification and classical AOI tools (V2 backlog).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M12 - MQTT

**Goal.** Provide an MQTT message publisher for publishing inspection events to
a broker.

#### Delivered

- `MqttPublisher` implementing the existing `MessagePublisher` seam.
- Pluggable `MqttClient` protocol for the low-level broker protocol, keeping
  the publisher free of external dependencies and testable with a fake client.
- Connection lifecycle (`connect` / `disconnect` / `is_connected`).
- `publish(topic, payload, qos, retain)` with JSON payload serialization.
- All client failures translated to `CommsError`; publishing requires an
  established connection.

#### Explicitly out of scope

- Command subscription / command handling (broker -> station).
- Station-level event emission wiring (orchestration).
- Supervised defect classification and classical AOI tools (V2 backlog).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M13 - Minimal API + Dashboard

**Goal.** Expose a minimal HTTP API and dashboard over persisted inspection
results.

#### Delivered

- FastAPI application factory `create_app(repository)`.
- `GET /health` health check.
- `GET /api/v1/results` and `GET /api/v1/results/{inspection_id}` read-only
  results endpoints backed by the `ResultRepository` seam.
- Minimal dashboard page served at `/` that lists recent results.
- Integration regression tests via FastAPI `TestClient`.

#### Explicitly out of scope

- Full API surface (recipes, calibration, monitoring) - Milestone M15.
- WebSocket live channels - Milestone M15.
- Supervised defect classification and classical AOI tools (V2 backlog).

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M14 - Monitoring / Metrics / SPC / Health

**Goal.** Provide runtime telemetry, statistical process control, and component
health checks for the station.

#### Delivered

- `MetricsRegistry` for counters, gauges, and histograms with a JSON-serializable
  snapshot.
- `control_chart` SPC helper computing mean, standard deviation, control limits,
  and out-of-control detection.
- `HealthCheck` aggregating named component health probes.
- Unit tests for all monitoring modules.

#### Explicitly out of scope

- Full API + dashboard integration of metrics - Milestone M15.
- Alerting / notification delivery - later milestone.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M15 - Full API + Dashboard

**Goal.** Extend the minimal API into a fuller API surface with metrics,
component health, and a live WebSocket channel, plus a richer dashboard.

#### Delivered

- `GET /api/v1/metrics` runtime metrics snapshot.
- `GET /api/v1/health` component health status.
- `GET /ws/results` WebSocket live channel for inspection results.
- Dashboard page served at `/`.
- Integration tests covering the new endpoints.

#### Explicitly out of scope

- Calibration lifecycle endpoints - Milestone M16.
- Failure handling / buffering - Milestone M17.
- Deployment / containers - Milestone M18.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M16 - Calibration Lifecycle + Hot-Swap + Self-Test

**Goal.** Manage the calibration lifecycle: validate a calibration artifact,
hot-swap it into the active set, and self-test it before activation.

#### Delivered

- `CalibrationSelfTest` validating pixel size, image dimensions, and intrinsic
  matrix invertibility.
- `CalibrationManager` holding the active calibration per camera with atomic
  hot-swap; a failed self-test rejects the swap and leaves the previous
  calibration active.
- Unit tests covering self-test, activation, hot-swap, and rejection.

#### Explicitly out of scope

- Failure handling / buffering - Milestone M17.
- Deployment / containers - Milestone M18.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M17 - Failure Handling + Buffering + Regression/Performance Suites

**Goal.** Make persistence resilient to transient failures and add regression
and performance suites to guard the walking skeleton.

#### Delivered

- `ResultBuffer` bounded FIFO buffer for results awaiting persistence.
- `FailureHandler` retrying persistence with a bounded attempt count and
  buffering results that still fail; `flush` retries buffered results.
- `tests/regression/` end-to-end regression suite.
- `tests/performance/` throughput smoke tests.
- Unit tests for buffering and failure handling.

#### Explicitly out of scope

- Deployment / containers / edge - Milestone M18.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M18 - Deployment / Containers / Edge

**Goal.** Containerize the AdaptiveVision API and ship an edge observability
stack (Prometheus + Grafana) so the station can be deployed and monitored on
edge hardware.

#### Delivered

- `deploy/Dockerfile` - slim Python 3.11 runtime image serving the API.
- `deploy/.dockerignore` - excludes non-runtime artifacts from the build.
- `deploy/docker-compose.yml` - API + Prometheus + Grafana stack.
- `deploy/prometheus.yml` - scrapes the API `/metrics` endpoint.
- `deploy/grafana/provisioning/` - datasource + dashboard auto-provisioning
  (kept as Grafana's required `provisioning/dashboards/` +
  `provisioning/datasources/` subdirectories - this is a hard constraint of
  Grafana's file-provisioning scanner, not sprawl).
- `deploy/grafana/adaptivevision.json` - edge dashboard.
- `deploy/deploy_edge.sh` - one-command edge deployment helper.
- `adaptivevision.monitoring.prometheus` - Prometheus text exposition renderer.
- `GET /metrics` endpoint serving Prometheus text format.
- `uvicorn` runtime dependency for the containerized API server.
- Unit tests for the renderer and integration tests for `/metrics`.

#### Explicitly out of scope

- Full station orchestration in containers - the station runs on the edge host;
  only the API + observability stack are containerized here.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M19 - Quality Intelligence & Deployment Observability

**Goal.** Add historical-defect retrieval, an evidence-based local-LLM
advisory layer, deployment-cost profiling, and a deterministic Pareto
deployment recommender, extending the existing architecture rather than
duplicating it.

#### Delivered

- `RetrievalIndex`, `AdvisoryEngine`, and `AdvisoryRepository` seams on
  `common/interfaces.py`; `RetrievalError` and `AdvisoryError` on
  `common/errors.py`; `RetrievalMatch`, `InspectionEvidence`, and
  `AdvisoryReport` value objects on `common/result.py`.
- `retrieval/faiss_index.py` - `FaissRetrievalIndex`, a FAISS-backed
  `RetrievalIndex` with L2/inner-product/cosine metrics, a versioned JSON
  metadata sidecar, and dimension/finite-value/embedding-configuration
  validation.
- `advisory/` - `OllamaAdvisoryEngine` (Pydantic-validated structured output,
  retry-then-deterministic-fallback, never raises), and a `pipeline` module
  (`build_evidence`/`advise`) that reads an already-final `Decision`'s
  severity read-only and rejects any advisory response that changes it.
- `inference/profiling.py` - `benchmark_latency`, measuring p50/p95 latency
  and throughput against any loaded `InferenceEngine` (complements
  `training/benchmark/cost.py`'s pre-export PyTorch profiling by measuring
  the actual deployed engine).
- `deployment/profiles.py` - `DeploymentProfile`, a deterministic Pareto
  frontier and constraint-based `recommend`/`explain_recommendation` (no LLM
  involved), reading a one-way, versioned JSON artifact produced by the new
  `training/benchmark/deployment_export.py` - production never depends on
  training code or the sweep database directly.
- `AdvisoryRecord` / `SqliteAdvisoryRepository` - a second, independent
  persistence table alongside `InspectionRecord`, following the same
  session-factory/exception-wrapping shape as `SqliteResultRepository`.
- Two new API routers (`/api/v1/advisory`, `/api/v1/deployment`), following
  the existing `get_<seam>()` + `dependency_overrides` composition-root
  pattern; both are optional and degrade gracefully (advisory routes are not
  registered at all without a repository; deployment profiles default to an
  empty tuple). The embedded dashboard page shows advisory info per row when
  available.
- `training/dashboard_app.py`'s existing upload-and-score workflow now also
  produces an advisory report per scored image (retrieval is not wired in
  here - see Explicitly out of scope).
- Unit/integration tests for every module above, following existing
  conventions (constructor-injected fakes, in-memory SQLite, `TestClient`).

#### Explicitly out of scope

- FAISS retrieval is not wired into any live inference path. The deployed
  PaDiM ONNX contract (`training/benchmark/export.py`'s `ProductionExport`)
  outputs only a calibrated scalar score, not an embedding vector - there is
  currently nothing to search against without changing that contract, which
  this milestone deliberately did not touch. `FaissRetrievalIndex` itself is
  complete and tested in isolation.
  **Update (2026-08-27, no dedicated milestone doc at the time):** the ONNX
  export contract was extended to a 3-output
  `(score, embedding, patch_features)` tuple, and `training/dashboard_app.py`
  now wires FAISS retrieval + the advisory pipeline into its live
  upload-and-score request path when a model with an embedding output is
  configured. This bullet is out of date for that entry point; it remains
  accurate for the *production* `src/adaptivevision/app`/`api` composition
  roots, which still don't touch retrieval at all (see M20).
- A live Ollama server was not available to exercise the real-LLM code path
  end-to-end; only the fallback path and the Pydantic-validated success path
  (via an injected fake client) were tested.
- The production Streamlit dashboard's dependencies (`streamlit`, `plotly`,
  `streamlit_autorefresh`) remain undeclared, a pre-existing gap this
  milestone did not close beyond documenting it.
- ONNX export/validation was not duplicated - `training/benchmark/export.py`
  already exports, calibrates, and verifies through the real production
  `OnnxInferenceEngine` + `ThresholdAnomalyDetector` path.

#### Acceptance criteria

`pip install -e .`, `pytest`, `ruff check .`, `black --check .`, and
`mypy src` pass from the `adaptivevision` Conda environment.

---

### Milestone M20 - Wire the M9/M10 Inspection Chain Into the Live Station

**Goal.** A full connectivity audit (every package under `src/adaptivevision`
traced against the two real composition roots, `app.app.build_station` and
`api.app.create_app`) found that M9's anomaly detector and M10's decision
policy were fully built and unit-tested but never once constructed by
`build_station()`. The result: `scripts/run_station.py` - the actual station
entry point - always returned `Verdict.PASS`, unconditionally, regardless of
configuration, because `InspectionPipeline._decide()`'s fallback rule only
fires when `partials` is empty, and it always was. This milestone wires that
chain in for real, fixes two genuine bugs the audit and the wiring work
surfaced, and proves the fix with real MVTec images end-to-end - not just that
the code no longer raises.

#### Delivered

- `app.app.build_recipe`/`build_anomaly_detector`/`build_decision_policy`,
  wired into `build_station()` exactly like the existing optional
  calibration/alignment stages: config-gated (`MODEL_PATH`,
  `DEFAULT_RECIPE_ID`), skeleton-without-hardware stays true when unset.
- `decision.policy.DecisionPolicy` now honors the *full* declared contract
  `recipe.model.DecisionPolicy` always promised ("the fields are the stable
  contract the decision engine will consume") but M10 never actually
  consumed: `fail_severity` (configurable, was hardcoded to
  MAJOR/CRITICAL) and `max_defects` (previously unused entirely). New
  `DecisionPolicy.from_recipe(recipe)` factory bridges the two. The one
  recipe field that still doesn't map onto this engine,
  `review_on_anomaly`, is honored one level down instead - see next bullet -
  since it's a property of what severity an anomaly *becomes*, not of how an
  already-built defect list is judged.
- `inspection.anomaly.detector.ThresholdAnomalyDetector` gained
  `anomalous_severity` (MINOR routes to REVIEW, MAJOR - the previous
  hardcoded behavior - routes to FAIL), and now actually works on a real
  camera frame: it never cast the image to the model's float32 dtype, so any
  non-float input (every real camera, uint8) failed inference outright; it
  also only ever handled a 2D (grayscale) frame, not a 3D (H, W, C) color
  one. Both are fixed - discovered by literally running the wired pipeline
  against real image files, not by inspection.
- `preprocessing.operators.resize_to(height, width)` - a dependency-free
  (no OpenCV in the production package) nearest-neighbor resize step, wired
  into `build_preprocessor()` via `MODEL_INPUT_HEIGHT`/`MODEL_INPUT_WIDTH`,
  so a configured model always receives frames matching its fixed input
  contract regardless of camera resolution.
- **Real bug, unrelated to this milestone's own code, found while verifying
  it**: `config.loader.load_config()`'s docstring has always claimed it
  "Defaults to `.env` in the current directory," but the implementation only
  ever read a `.env` file when a caller passed `env_file=` explicitly -
  which neither `scripts/run_station.py` nor `scripts/run_api.py` do. A
  committed `.env.example`, copied to `.env` exactly as its own header says
  to, was never actually read by the real entry points, for the entire life
  of the project. Fixed: `env_file` now defaults to `Path(".env")`.
- **Second real, pre-existing bug**: `deploy/docker/Dockerfile`'s `CMD` ran
  `uvicorn adaptivevision.api:create_app --factory`, which calls the factory
  with zero arguments - but `create_app()`'s `repository` parameter has no
  default. The container built successfully and crashed on every start.
  Fixed to run `scripts/run_api.py` instead, the one place that actually
  builds the repository/advisory/deployment-profiles composition
  `create_app()` needs (not verified against a real `docker build`/`run` -
  no Docker available on this machine - but the corrected command is the
  exact one already exercised by the full test suite via `create_app`'s own
  call sites).
- `recipes/demo-bottle.json` and `.env.example` updated so
  `scripts/run_station.py` exercises real ONNX inference out of the box
  (`models/mvtec_bottle.onnx`, grayscale, matches what the null camera can
  produce without real hardware).
- `tests/e2e/test_real_inspection.py` - proves the wired chain gives the
  *correct* verdict against real MVTec bottle photos (not synthetic data):
  good parts score 0.44-0.60 and PASS, every defect category tested
  (broken_large, broken_small, contamination) scores 1.0 and FAILs, using
  `models/patchcore_dinov2_mvtec_bottle.onnx` (the properly-benchmarked
  PatchCore+DINOv2 export from `training/benchmark/`, not the earlier
  autoencoder). Skipped when that model/dataset aren't present locally.
- `tests/e2e/test_run_station.py` updated to run with `cwd` pointed at an
  empty temp directory - it asserts the *unconfigured* skeleton still PASSes
  unconditionally, which is no longer true when a real `.env` with a model
  configured sits in the repo root (exactly the `.env.example` default this
  milestone ships), so the test must not depend on ambient repo state.
- `pyproject.toml`: `onnxruntime` is now declared (optional `inference`
  group) since it backs a real, if optional, production code path -
  previously left undeclared as "training-only," which stopped being true
  the moment `build_anomaly_detector` could construct an
  `OnnxInferenceEngine`. `opencv-python-headless` added to `dev`, used only
  by `test_real_inspection.py`'s real-image fixtures.

#### Explicitly out of scope

Found by the same audit, not addressed here - each is a real, tested,
disconnected module, not a design gap:

- `communication/mqtt/publisher.py` and `communication/plc/modbus.py`:
  neither is ever called from `orchestration/` or `app/`. No "publish result"
  or "write PLC coil" call exists anywhere outside their own unit tests.
- `orchestration/watchdog.py::CycleWatchdog`: constructed and injected into
  `StationController`, but nothing ever calls `.check()` - both the class's
  and `station.py`'s own docstrings claim otherwise.
- `orchestration/failure.py::FailureHandler`: `persistence/integration.py`'s
  real `on_result` hook still does its own bare log-and-drop instead of
  using it, despite M17's stated goal being persistence resilience.
- `monitoring/metrics.py::MetricsRegistry` and `monitoring/health.py`:
  wired to the API (`/metrics`, `/api/v1/health`), but nothing in production
  ever calls `.increment()`/`.set_gauge()`/`.register()` - both endpoints
  are live but permanently empty. `deploy/grafana/dashboards/adaptivevision.json`
  references metric names (`adaptivevision_inspections_total`, etc.) that
  don't exist anywhere in the code as a result.
- `persistence/image_store.py::LocalImageStore`: never passed to
  `make_persistence_handler()`, so `image_refs` are persisted as bare frame
  IDs with no actual image bytes archived.
- `inspection/metrology/inspector.py::MetrologyInspector`: still not wired.
  Unlike anomaly detection, it needs a real `MeasurementSource` - concrete
  dimensional-measurement extraction code that doesn't exist yet anywhere in
  the repo - and fabricating one to force a wire-up would silently produce
  wrong measurements, which is worse than leaving it correctly optional.

#### Acceptance criteria

`pytest`, `ruff check .`, `black --check .`, and `mypy src scripts` pass from
the `adaptivevision` Conda environment (375 tests, 95.53% coverage).
`python scripts/run_station.py` with `.env.example` copied to `.env`
produces a real, model-driven verdict (`fail`, `anomaly_score=1.0`, against
the null camera's blank frame) instead of the previous unconditional `pass`.
