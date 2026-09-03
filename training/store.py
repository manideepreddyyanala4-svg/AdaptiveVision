"""Persistence for the sweep: the SQLite results row, raw-prediction archives, and model checkpoints.

Three flavors of "save the sweep's output," kept together because they're
the same concern at different granularities:

1. The SQLite ``runs`` table (:class:`RunRow`) -- one row per
   ``(method, regime, config, seed)``, crash-safe resumable, backfillable.
2. Per-run prediction archives (:func:`save_artifact`/:func:`load_artifact`)
   -- raw scores/labels/anomaly maps, so the dashboard, the ensembler, and
   any metric invented later never have to re-fit a model to get them.
3. Per-run model checkpoints (:func:`save_checkpoint`/:func:`load_checkpoint`)
   -- the fitted ``torch.nn.Module`` itself, so the deployment-cost pass can
   load it back and time a real forward pass instead of re-fitting.

Replaces the old JSONL file (``append_row``/``load_completed``,
last-write-wins by re-scanning the whole file on every launch) with a real
table, because two things the sweep needs don't fit an append-only file:

* **Crash-safe resume at run granularity.** A row is inserted as
  ``status="running"`` before its fit even starts, then updated in place to
  ``"ok"``/``"failed"`` when it finishes. A row still ``"running"`` on the next
  launch is unambiguously a crash victim -- delete and retry, no last-write-wins
  guessing.
* **Backfilling columns after the fact.** The deployment-cost pass and the
  new-metrics pass (AUPIMO/PG2/PB2, read from the already-saved ``.npz``
  artifacts) both need to add columns to a row that already exists from the
  accuracy sweep, without re-running anything. That is an ``UPDATE``, which a
  JSONL file cannot do without rewriting itself.

Schema evolution is ``create_all()``-only, matching the only precedent in
this repo (``src/adaptivevision/storage.py``) -- no Alembic. A ``run_id``
bakes in :data:`SCHEMA_VERSION`, so a version bump orphans old rows instead
of colliding with them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    event,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from torch import nn

# -----------------------------------------------------------------------------
# The SQLite results store
# -----------------------------------------------------------------------------

#: Bump this when the row schema changes shape in a way that would make an old
#: row's columns mean something different than a new one's. Baked into every
#: run_id, so old rows simply stop matching -- no migration needed.
SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    """Declarative base for the benchmark results store."""


class RunRow(Base):
    """One ``(method, regime, config, seed, defect_kind, ...)`` run.

    The row *is* the record -- there is no separate domain object to map to,
    unlike the production ``InspectionRecord``/``ResultRepository`` boundary.
    """

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    schema_version: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # "running" | "ok" | "failed"
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    host: Mapped[str] = mapped_column(String(128), default="")
    torch_version: Mapped[str] = mapped_column(String(32), default="")

    # Identity -- mirrors training.sweep's _base_row plus the regime-specific extras.
    regime: Mapped[str] = mapped_column(String(32), index=True)
    method: Mapped[str] = mapped_column(String(64), index=True)
    family: Mapped[str] = mapped_column(String(32), index=True)
    backend: Mapped[str] = mapped_column(String(16))
    config: Mapped[str] = mapped_column(String(64), index=True)
    dataset: Mapped[str] = mapped_column(String(32), index=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seed: Mapped[int] = mapped_column(Integer, index=True)
    defect_kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    single_seed: Mapped[bool] = mapped_column(Boolean, default=False)
    severstal_target_prevalence: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_noise_caveat: Mapped[bool] = mapped_column(Boolean, default=False)
    test_prevalence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Accuracy metrics (ImageMetrics.as_dict() + pg2/pb2), all nullable --
    # absent until the accuracy sweep (or, for pg2/pb2, the metrics backfill
    # pass) has produced them.
    auroc: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    balanced_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    balanced_error_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    fpr_at_95tpr: Mapped[float | None] = mapped_column(Float, nullable=True)
    fnr_at_1fpr: Mapped[float | None] = mapped_column(Float, nullable=True)
    pg2: Mapped[float | None] = mapped_column(Float, nullable=True)
    pb2: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_normal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_anomalous: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Pixel metrics (PixelMetrics.as_dict() + aupimo).
    pixel_auroc: Mapped[float | None] = mapped_column(Float, nullable=True)
    pixel_average_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    aupro: Mapped[float | None] = mapped_column(Float, nullable=True)
    aupimo: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_images: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Deployment-cost metrics, backfilled from a checkpoint load by
    # training.evaluate's cost pass -- absent until that pass has visited
    # this run_id.
    inference_latency_ms_p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_latency_ms_p95: Mapped[float | None] = mapped_column(Float, nullable=True)
    throughput_fps_bs1: Mapped[float | None] = mapped_column(Float, nullable=True)
    throughput_fps_bs16: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_params_millions: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_gpu_memory_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_wall_clock_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Existing sweep-timing columns.
    n_test: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    ms_per_image: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_vram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_fit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fit_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    elapsed_total_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Regime-specific extras.
    multiclass_family: Mapped[str | None] = mapped_column(String(32), nullable=True)
    multiclass_categories: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_shot: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Failure.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    traceback: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Non-hashed provenance: sweep-wide knobs that describe "the study", not
    # this run's identity (see compute_run_id's docstring for why they're
    # excluded from the hash).
    run_options_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


def compute_run_id(
    method: str,
    regime: str,
    config_key: str,
    seed: int,
    defect_kind: str | None = None,
    severstal_target_prevalence: float | None = None,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """A stable id for one run's identity.

    Two calls with the same identity always produce the same id, regardless of
    dict/kwarg order -- ``json.dumps(..., sort_keys=True)`` before hashing.

    Args:
        method: Registry method name.
        regime: ``"oneclass"``, ``"multiclass"``, or ``"fewshot{k}"``.
        config_key: The individual category's key for one-class/few-shot
            (e.g. ``"mvtec/bottle"``), or the family name for multi-class
            (e.g. ``"mvtec"``) -- matches ``run_multiclass``'s one-row-per-
            category yield sharing a single fit.
        seed: The run's seed.
        defect_kind: ``"structural"``/``"logical"`` for an MVTec LOCO
            breakdown row, else ``None``. Real hash input, not an afterthought
            -- without it, LOCO's combined + 2 breakdown rows would collide
            onto one run_id.
        severstal_target_prevalence: The prevalence a Severstal run was
            subsampled to, or ``None`` for an unsubsampled run.
        schema_version: Baked into the hash so a schema bump can't silently
            collide with an old row's differently-shaped columns.

    Returns:
        A 16-character hex id.
    """
    payload = {
        "schema_version": schema_version,
        "method": method,
        "regime": regime,
        "config": config_key,
        "seed": seed,
        "defect_kind": defect_kind,
        "severstal_target_prevalence": severstal_target_prevalence,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine, enabling WAL so a live reader (the
    dashboard) never blocks on this process's writes.
    """
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def open_database(path: str | Path) -> tuple[Engine, sessionmaker[Session]]:
    """Open (and initialize) the writer connection to ``path``.

    Creates the parent directory first -- SQLite cannot create the file
    inside a directory that doesn't exist yet, unlike the old JSONL
    ``append_row``, which handled this itself.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = build_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False, future=True)


def open_readonly(path: str | Path) -> tuple[Engine, sessionmaker[Session]]:
    """Open a read-only connection to ``path``, for the dashboard.

    Connects via a ``mode=ro`` URI so this process can never write, even by
    mistake. Does not call ``create_all`` -- the writer already owns schema
    creation, and a reader shouldn't be able to create the file if it's
    missing.
    """
    engine = create_engine(f"sqlite:///file:{Path(path)}?mode=ro&uri=true", future=True)
    return engine, sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextlib.contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """A transactional session, committed on success and rolled back on error."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def start_run(session_factory: sessionmaker[Session], run_id: str, identity: dict[str, Any]) -> None:
    """Insert a placeholder row before a fit starts.

    A row left in this state (``status="running"``) on the next launch is a
    crash victim -- see :func:`reset_incomplete`.

    Upserts rather than raw-inserts: a multi-class job that's pending because
    one of its sibling categories is crash-debris still re-registers *every*
    category's row (the whole family refits together, see run_multiclass),
    including ones already ``status="ok"`` from before the crash. Deleting
    any existing row for this run_id first is what makes that legal instead
    of hitting the UNIQUE constraint.
    """
    with session_scope(session_factory) as session:
        session.execute(delete(RunRow).where(RunRow.run_id == run_id))
        session.add(
            RunRow(
                run_id=run_id,
                schema_version=SCHEMA_VERSION,
                status="running",
                started_at=datetime.now(UTC),
                **identity,
            )
        )


def finish_run(session_factory: sessionmaker[Session], run_id: str, row: dict[str, Any]) -> None:
    """Update the same row a matching :func:`start_run` inserted.

    Args:
        session_factory: The store's session factory.
        run_id: The run being completed.
        row: Every column to set -- must include ``status`` (``"ok"`` or
            ``"failed"``) and whatever metric/error columns apply.
    """
    with session_scope(session_factory) as session:
        existing = session.scalar(select(RunRow).where(RunRow.run_id == run_id))
        if existing is None:
            msg = f"finish_run called for {run_id!r} with no matching start_run row"
            raise RuntimeError(msg)
        for key, value in row.items():
            setattr(existing, key, value)
        existing.finished_at = datetime.now(UTC)


def update_columns(session_factory: sessionmaker[Session], run_id: str, columns: dict[str, Any]) -> bool:
    """Backfill columns onto an existing row, e.g. cost metrics or new-metric columns.

    Used by the deployment-cost pass and the metrics-backfill pass, both of
    which add data to a row the accuracy sweep already wrote, without
    re-running anything.

    Returns:
        Whether a matching row was found and updated.
    """
    with session_scope(session_factory) as session:
        existing = session.scalar(select(RunRow).where(RunRow.run_id == run_id))
        if existing is None:
            return False
        for key, value in columns.items():
            setattr(existing, key, value)
        return True


def reset_incomplete(
    session_factory: sessionmaker[Session], scope_ids: set[str], force: bool = False
) -> int:
    """Delete crash-debris (or, with ``force``, everything) within ``scope_ids``.

    Scoped to the current plan's run_id set, never table-wide, so two
    differently-scoped invocations can't delete each other's in-flight rows.
    ``status="failed"`` is treated the same as ``"running"`` here -- a retry
    would violate the run_id UNIQUE constraint otherwise, and this matches the
    old JSONL behavior where failed and never-attempted were indistinguishable
    and always retried.

    Returns:
        The number of rows deleted.
    """
    if not scope_ids:
        return 0
    with session_scope(session_factory) as session:
        condition = RunRow.run_id.in_(scope_ids) if force else (
            RunRow.run_id.in_(scope_ids) & RunRow.status.in_(["running", "failed"])
        )
        result = session.execute(delete(RunRow).where(condition))
        return result.rowcount or 0


def completed_run_ids(session_factory: sessionmaker[Session], scope_ids: set[str]) -> set[str]:
    """Every run_id in ``scope_ids`` that already has a ``status="ok"`` row."""
    if not scope_ids:
        return set()
    with session_scope(session_factory) as session:
        rows = session.scalars(
            select(RunRow.run_id).where(RunRow.run_id.in_(scope_ids), RunRow.status == "ok")
        )
        return set(rows)


# -----------------------------------------------------------------------------
# Per-run prediction archives
# -----------------------------------------------------------------------------
#
# The sweep is the expensive part, so it must only ever run once. Every run
# writes its raw scores (and, where produced, its anomaly maps) to a
# compressed archive keyed by regime/method/config. Everything downstream
# reads those instead of re-fitting: the dashboard draws ROC curves and score
# histograms from them, the ensembler fuses methods that were never run
# together, and any metric invented later can be recomputed without touching
# a GPU.
#
# Maps are stored at a reduced resolution and as float16. Full-resolution
# float32 maps for 29 configurations across a zoo this size would run to
# hundreds of gigabytes, and nothing downstream needs that precision -- the
# pixel metrics resize to a common grid anyway.

#: Anomaly maps are stored on a grid no larger than this on the long side.
MAP_STORE_SIZE = 256


@dataclass(frozen=True)
class RunArtifact:
    """Raw predictions for one ``(regime, method, config)`` run.

    Attributes:
        scores: ``(N,)`` image-level anomaly scores.
        labels: ``(N,)`` ground-truth anomaly labels.
        paths: ``(N,)`` source image paths, as strings, for the gallery.
        maps: ``(N, h, w)`` anomaly maps, or ``None`` if the method produced none.
    """

    scores: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    maps: np.ndarray | None

    def __len__(self) -> int:
        """Number of scored images."""
        return int(self.scores.shape[0])


def artifact_path(root: Path, regime: str, method: str, config_key: str, seed: int) -> Path:
    """Location of one run's archive.

    Keyed by seed too: with the 3-seed repeats, three separate fits produce
    three separate score/label/map sets for the same (regime, method,
    config) -- without the seed in the filename, the second and third fit
    would each silently overwrite the previous seed's saved archive.
    """
    slug = config_key.replace("/", "_")
    return root / regime / f"{method}__{slug}__seed{seed}.npz"


def save_artifact(
    path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    maps: np.ndarray | None = None,
) -> None:
    """Write one run's predictions.

    Args:
        path: Destination ``.npz``.
        scores: Image-level anomaly scores.
        labels: Ground-truth anomaly labels.
        paths: Source image paths, in the same order.
        maps: Optional anomaly maps; downsampled and cast to float16 here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=bool),
        "paths": np.asarray(paths, dtype=object),
    }
    if maps is not None:
        payload["maps"] = _shrink_maps(np.asarray(maps)).astype(np.float16)
    np.savez_compressed(path, **payload)


def load_artifact(path: Path) -> RunArtifact | None:
    """Read one run's predictions, or ``None`` if the archive is absent."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return RunArtifact(
            scores=data["scores"],
            labels=data["labels"],
            paths=data["paths"],
            maps=data["maps"].astype(np.float32) if "maps" in data else None,
        )


def _shrink_maps(maps: np.ndarray) -> np.ndarray:
    """Downsample maps so the long side is at most :data:`MAP_STORE_SIZE`.

    Uses OpenCV area interpolation, which preserves peak location better than
    naive striding -- and peak location is the whole point of a heatmap.
    """
    if maps.ndim != 3:
        return maps
    height, width = int(maps.shape[1]), int(maps.shape[2])
    longest = max(height, width)
    if longest <= MAP_STORE_SIZE:
        return maps

    import cv2

    scale = MAP_STORE_SIZE / longest
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.stack(
        [cv2.resize(m.astype(np.float32), target, interpolation=cv2.INTER_AREA) for m in maps]
    )


# -----------------------------------------------------------------------------
# Per-run model checkpoints
# -----------------------------------------------------------------------------
#
# Every job's scorer -- whether it is a PatchCore memory bank, a PaDiM
# Gaussian, a DFM PCA basis, or a trained Dinomaly decoder -- is a plain
# torch.nn.Module with no unpicklable state (no open handles, no stored
# hooks, no live CUDA streams). That means a whole-object save round-trips
# cleanly for every method in the zoo without any per-family bespoke
# serialization: the same two functions here work for all of them.
#
# Saving happens once per fit, right after a method's fit() returns and
# before the scorer is used for scoring or goes out of scope -- see
# training.sweep. This is what lets the deployment-cost pass measure real
# inference latency/VRAM/params by loading a checkpoint and timing its
# forward pass, instead of re-fitting just to get a model to time.


def checkpoint_path(root: Path, regime: str, method: str, config_key: str) -> Path:
    """Location of one run's saved model.

    Mirrors :func:`artifact_path`'s ``{regime}/{method}__{slug}`` layout so
    the two persisted-per-run archives (predictions, model) stay easy to
    correlate on disk.
    """
    slug = config_key.replace("/", "_")
    return root / regime / f"{method}__{slug}.pt"


def save_checkpoint(scorer: nn.Module, path: Path) -> None:
    """Write a fitted scorer to disk.

    Args:
        scorer: The fitted model, straight out of a method's ``fit()``.
        path: Destination ``.pt``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scorer, path)


def load_checkpoint(path: Path, device: str = "cuda") -> nn.Module | None:
    """Read a saved scorer back, or ``None`` if the checkpoint is absent.

    Args:
        path: Checkpoint written by :func:`save_checkpoint`.
        device: Device to map the restored tensors onto.

    Returns:
        The restored scorer in eval mode, ready to score or time, or ``None``
        if nothing was saved at ``path`` (e.g. the run failed before fitting).
    """
    if not path.exists():
        return None
    scorer = torch.load(path, map_location=device, weights_only=False)
    scorer.eval()
    return scorer
