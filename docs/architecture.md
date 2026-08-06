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

Per-milestone implementation notes live in `docs/milestones/`.
