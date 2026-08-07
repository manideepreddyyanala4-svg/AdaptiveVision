# AdaptiveVision

**Industrial 2D/3D Vision, AOI & Metrology Edge Platform.**

An edge-deployed industrial inspection station: it acquires images of a part,
corrects optics, aligns the part to a golden reference, runs classical
metrology and AI anomaly detection, fuses a PASS / FAIL / REVIEW verdict, and
integrates with the factory floor over Modbus TCP and MQTT.

The system design is frozen in **Architecture Specification v1.0** and built
according to **Implementation Roadmap v1.0** (see `docs/`).

> **Status:** Milestone **M1 — Domain Contracts & Interfaces** complete.
> The shared value types, enums, error taxonomy, geometry, timing, and the eight
> abstraction seams are in place under `adaptivevision.common`. These are
> contracts only; no inspection behaviour exists yet (that begins at M3).

## Requirements

- Python **3.11+**

## Quick start

```bash
# 1. Install the package with dev tooling (editable)
make install            # or: pip install -e ".[dev]"

# 2. Boot the station shell (M0: logs one structured line and exits)
make run                # or: python scripts/run_station.py

# 3. Run the full local quality gate (mirrors CI)
make check              # ruff + format check + mypy + pytest
```

Example boot output (a single JSON line):

```json
{"timestamp": "2026-01-01T00:00:00.000+00:00", "level": "INFO", "logger": "adaptivevision.boot", "message": "AdaptiveVision station starting", "correlation_id": "boot-ab12cd34ef56", "module": "run_station", "function": "main", "line": 40, "version": "0.0.0", "milestone": "M0", "state": "INIT"}
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

## Repository layout

```
src/adaptivevision/     # application package (frozen tree, Spec v1.0 §7.1)
scripts/                # operational entrypoints (run_station)
tests/                  # unit / integration / e2e / performance
configs/ recipes/       # station config + product recipes (M2)
calibration/ models/    # calibration artifacts + ONNX models (M5/M8)
deploy/                 # containerization + observability (M18)
docs/                   # architecture notes + milestone records
```

Most sub-packages are intentionally empty at M0 and are implemented in their
scheduled milestone (noted in each package docstring and `docs/milestones/`).
