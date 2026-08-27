# AdaptiveVision

**Industrial 2D/3D Vision, AOI & Metrology Edge Platform.**

An edge-deployed industrial inspection station: it acquires images of a part,
corrects optics, aligns the part to a golden reference, runs classical
metrology and AI anomaly detection, fuses a PASS / FAIL / REVIEW verdict, and
integrates with the factory floor over Modbus TCP and MQTT.

The system design is frozen in **Architecture Specification v1.0** and built
according to **Implementation Roadmap v1.0** (see `docs/`).

> **Status:** Milestones **M0–M19** complete.
> The platform spans the full roadmap: domain contracts (M1), configuration &
> recipes (M2), walking skeleton (M3), persistence (M4), calibration (M5/M16),
> inference (M8), anomaly detection (M9), decision fusion (M10), PLC/MQTT
> integration (M11/M12), HTTP API + dashboard (M13/M15), monitoring/SPC/health
> (M14), failure handling & buffering (M17), containerized edge deployment
> with Prometheus + Grafana observability (M18), and historical-defect
> retrieval + LLM advisory + deployment observability (M19).




## Requirements

- Python **3.11+**

## Quick start

```bash
# 1. Install the package with dev tooling (editable)
make install            # or: pip install -e ".[dev]"

# 2. Boot the station (M3: boots, runs a demo inspection, shuts down)
make run                # or: python scripts/run_station.py


# 3. Run the full local quality gate (mirrors CI)
make check              # ruff + format check + mypy + pytest
```

Example boot output (a single JSON line):

```json
{"timestamp": "2026-01-01T00:00:00.000+00:00", "level": "INFO", "logger": "adaptivevision.boot", "message": "AdaptiveVision station starting", "correlation_id": "boot-ab12cd34ef56", "module": "run_station", "function": "main", "line": 40, "version": "0.0.0", "milestone": "M2", "state": "INIT", "station_id": "station-01"}

```

## Developer commands

| Command | Purpose |
|---|---|
| `make lint` | Ruff lint |
| `make format` | Ruff auto-format |
| `make format-check` | Ruff format check (CI gate) |
| `make typecheck` | mypy (strict, `src` + `scripts`) |
| `make test` | pytest + coverage |
| `make check` | All of the above |

## Quality intelligence & deployment (M19)

Two optional, lazily-imported capabilities layer on top of the core
platform - the station runs identically without either:

- **Historical-defect retrieval** (`adaptivevision.retrieval`, backed by
  FAISS) and **LLM advisory** (`adaptivevision.advisory`, backed by Ollama)
  explain evidence; they never set or override the deterministic severity
  computed by `adaptivevision.decision`. Enable with:
  `pip install -e ".[intelligence]"`. Without it, advisory calls fall back to
  a deterministic report derived from evidence alone - nothing crashes.
- **Deployment profiling & recommendation** (`adaptivevision.deployment`)
  reads a versioned JSON artifact exported from the research sweep
  (`python training/benchmark/deployment_export.py`) and answers "which
  model should I deploy under my latency/accuracy/size constraints" via
  `GET /api/v1/deployment/recommendation` - always a deterministic pick, no
  LLM involved.

See `docs/milestones/M19.md` for what was delivered. FAISS retrieval is now
wired into a live inference path (`training/dashboard_app.py`'s upload-and-
score flow), grounding the LLM advisory report in real historical defect
matches instead of text-only evidence.

## Repository layout

```
src/adaptivevision/     # application package (frozen tree, Spec v1.0 §7.1)
scripts/                # operational entrypoints (run_station)
tests/                  # unit / integration / e2e / performance
models/                 # exported ONNX models (M5/M8)
deploy/                 # containerization + observability (M18)
docs/                   # architecture notes + milestone records
```

Every package under `src/adaptivevision/` is implemented as of M19; two
narrow, explicitly-documented exceptions remain planned for later milestones
(`inspection/classical/`, `communication/digital_io/` - see their own
docstrings). `configs/`, `recipes/`, and `calibration/` are created at
runtime by the code that populates them and are not checked in empty.
