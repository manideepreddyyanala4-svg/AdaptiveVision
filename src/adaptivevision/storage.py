"""Record stage: save it, trace it, report it.

The SQLite database layer, the ORM schema, the repository implementations
that map domain objects to it, bounded local image archival, the persistence
handler wired into the inspection critical path, and traceability / MES
(Manufacturing Execution System) event-payload export.

Persistence must stay off the inspection critical path: a database failure
must never crash or block the inspection loop, so :class:`PersistenceHandler`
logs and swallows failures rather than propagating them. Domain objects
(:class:`~adaptivevision.common.InspectionResult`) never import or depend on
SQLAlchemy types, and the ORM models never leak into the domain layer -- the
mapping between the two happens only in :func:`_to_record`/:func:`_from_record`.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Engine, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from adaptivevision.common import (
    AdaptiveVisionError,
    AdvisoryReport,
    AdvisoryRepository,
    Defect,
    DefectMeasurement,
    InspectionEvidence,
    InspectionResult,
    Measurement,
    ResultRepository,
    Verdict,
)

logger = logging.getLogger("adaptivevision.persistence")

# =============================================================================
# SQLite database layer
#
# Owns the SQLAlchemy engine and session factory for the local edge database.
# Supports a normal, file-backed SQLite database for local operation and an
# in-memory SQLite database ("sqlite:///:memory:") for tests. Schema
# initialization is idempotent: init_db creates any missing tables via the
# ORM metadata and is safe to call repeatedly.
# =============================================================================

#: Default SQLite URL used when no explicit path is provided.
_DEFAULT_DB_PATH = "adaptivevision.db"


class Base(DeclarativeBase):
    """Declarative base for all persistence ORM models."""


def build_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the local edge database.

    Args:
        url: SQLAlchemy database URL. Defaults to a file-backed SQLite database
            named :data:`_DEFAULT_DB_PATH` in the current directory. Pass
            ``"sqlite:///:memory:"`` for an in-memory database (tests).

    Returns:
        A configured :class:`~sqlalchemy.Engine`.
    """
    if url is None:
        url = f"sqlite:///{_DEFAULT_DB_PATH}"
    return create_engine(url, future=True)


def init_db(engine: Engine) -> None:
    """Create all tables defined by the ORM metadata.

    Idempotent: existing tables are left untouched.

    Args:
        engine: The engine to initialize the schema on.
    """
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a configured session factory bound to ``engine``.

    Args:
        engine: The engine the sessions will use.

    Returns:
        A :class:`~sqlalchemy.orm.sessionmaker` producing
        :class:`~sqlalchemy.orm.Session` objects.
    """
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def open_database(
    path: str | Path | None = None,
) -> tuple[Engine, sessionmaker[Session]]:
    """Open (and initialize) the local edge database.

    Args:
        path: Optional filesystem path for the SQLite file. When ``None``, the
            default file-backed database is used. Pass ``":memory:"`` for an
            in-memory database.

    Returns:
        A tuple of ``(engine, session_factory)`` with the schema initialized.
    """
    if path == ":memory:":
        url = "sqlite:///:memory:"
    elif path is None:
        url = None
    else:
        url = f"sqlite:///{Path(path)}"
    engine = build_engine(url)
    init_db(engine)
    return engine, make_session_factory(engine)


@contextlib.contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional session scope.

    Yields a session that is committed on success and rolled back on error.

    Args:
        session_factory: The session factory to create sessions from.

    Yields:
        A :class:`~sqlalchemy.orm.Session` within a transaction.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# =============================================================================
# ORM schema
#
# These models are the *persistence* representation of an inspection result,
# deliberately kept completely separate from the domain models in common.py.
# Lineage fields (recipe / model / calibration versions) are stored as
# first-class columns so they can be queried directly; the richer, nested
# result data (measurements, defects, image references) is serialized to
# JSON columns via the domain to_dict/from_dict contract. AdvisoryRecord
# (Milestone M19) is a *second*, independent table -- it never modifies
# InspectionRecord, matching the read-only relation between an inspection
# and its (optional) advisory report.
# =============================================================================


class InspectionRecord(Base):
    """ORM model for a single persisted inspection result.

    Attributes:
        id: Surrogate primary key.
        inspection_id: Unique identifier of the inspection (traceable).
        part_id: Identifier of the inspected part.
        station_id: Identifier of the producing station.
        verdict: Final verdict (``pass`` / ``fail`` / ``review``).
        recipe_ver: Version of the active recipe.
        model_ver: Version of the anomaly model, if any.
        calib_ver: Version of the calibration applied.
        cycle_time_ms: End-to-end inspection time in milliseconds.
        timestamp_utc: Completion time (timezone-aware, UTC).
        measurements_json: Serialized measurements.
        defects_json: Serialized defects.
        anomaly_score: Overall anomaly score, if computed.
        image_refs_json: Serialized references to archived images.
        defect_measurements_json: Serialized heatmap-derived shape
            measurements (Milestone M21).
        defect_count: Number of measured defect regions (``0`` both when
            metrology found none and when it wasn't run -- the same
            can't-tell-apart convention already used by ``defects_json``'s
            empty-list default). A first-class column (rather than only
            nested in ``defect_measurements_json``) so a fab can query/trend
            it directly, matching how ``anomaly_score`` is already a
            queryable column rather than only living inside ``defects_json``.
        max_defect_area_um2: Largest single measured defect area, in square
            microns.
        defect_type: Morphology of the largest measured defect (``"scratch"``
            or ``"particle"`` -- see ``metrology.py``), or ``None`` when no
            defect was measured.
        drift_status: Sensor/illumination drift status in effect at the time
            of this inspection, or ``None`` when no drift detector was wired
            in (see ``drift.py``).
    """

    __tablename__ = "inspection_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    part_id: Mapped[str] = mapped_column(String(128), index=True)
    station_id: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[str] = mapped_column(String(16))
    recipe_ver: Mapped[str] = mapped_column(String(64))
    model_ver: Mapped[str] = mapped_column(String(64), default="")
    calib_ver: Mapped[str] = mapped_column(String(64), default="")
    cycle_time_ms: Mapped[float] = mapped_column(Float)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    measurements_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    defects_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_refs_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    traceability_json: Mapped[str] = mapped_column(Text, default="")
    defect_measurements_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    defect_count: Mapped[int] = mapped_column(Integer, default=0)
    max_defect_area_um2: Mapped[float | None] = mapped_column(Float, nullable=True)
    defect_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    drift_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)


class AdvisoryRecord(Base):
    """ORM model for a single persisted advisory report (Milestone M19).

    Independent of :class:`InspectionRecord`: an inspection may have zero or
    one advisory report, linked only by ``inspection_id`` (no foreign-key
    constraint, matching this schema's JSON-column style of keeping the
    tables loosely coupled).

    Attributes:
        id: Surrogate primary key.
        inspection_id: Identifier of the inspection this report explains.
        evidence_json: Serialized :class:`~adaptivevision.common.InspectionEvidence`.
        report_json: Serialized :class:`~adaptivevision.common.AdvisoryReport`.
        created_at: Time the report was persisted (timezone-aware, UTC).
    """

    __tablename__ = "advisory_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


# =============================================================================
# Repositories
#
# Implements the ResultRepository/AdvisoryRepository seams against the local
# SQLite database, mapping domain objects to ORM records at this boundary.
# Storage failures surface as the base AdaptiveVisionError, matching the
# contract declared on the seams.
# =============================================================================


class SqliteResultRepository(ResultRepository):
    """A :class:`~adaptivevision.common.ResultRepository` backed by the local SQLite database.

    Args:
        session_factory: The session factory to use for database access.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Initialize the repository."""
        self._session_factory = session_factory

    def save_result(self, result: InspectionResult) -> None:
        """Persist an inspection result.

        Args:
            result: The inspection result to persist.

        Raises:
            AdaptiveVisionError: On storage failure.
        """
        record = _to_record(result)
        try:
            with self._session_factory() as session:
                session.add(record)
                session.commit()
        except Exception as exc:
            msg = f"Failed to persist inspection {result.inspection_id!r}: {exc}"
            raise AdaptiveVisionError(msg) from exc

    def get_result(self, inspection_id: str) -> InspectionResult | None:
        """Return the result with ``inspection_id``, or ``None`` if absent.

        Args:
            inspection_id: Identifier of the inspection to retrieve.

        Returns:
            The matching :class:`~adaptivevision.common.InspectionResult`, or
            ``None``.
        """
        try:
            with self._session_factory() as session:
                record = session.scalar(
                    select(InspectionRecord).where(InspectionRecord.inspection_id == inspection_id)
                )
        except Exception as exc:
            msg = f"Failed to query inspection {inspection_id!r}: {exc}"
            raise AdaptiveVisionError(msg) from exc
        return _from_record(record) if record is not None else None

    def list_results(self, *, limit: int = 100, offset: int = 0) -> tuple[InspectionResult, ...]:
        """Return a page of results ordered most-recent first.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            A tuple of :class:`~adaptivevision.common.InspectionResult`,
            most-recent first.
        """
        try:
            with self._session_factory() as session:
                records = session.scalars(
                    select(InspectionRecord)
                    .order_by(
                        InspectionRecord.timestamp_utc.desc(),
                        InspectionRecord.id.desc(),
                    )
                    .limit(limit)
                    .offset(offset)
                ).all()
        except Exception as exc:
            msg = "Failed to list inspection results"
            raise AdaptiveVisionError(msg) from exc
        return tuple(_from_record(r) for r in records)


def _to_record(result: InspectionResult) -> InspectionRecord:
    """Map a domain :class:`~adaptivevision.common.InspectionResult` to an ORM record."""
    areas = [m.area_um2 for m in result.defect_measurements]
    return InspectionRecord(
        inspection_id=result.inspection_id,
        part_id=result.part_id,
        station_id=result.station_id,
        verdict=result.verdict.value,
        recipe_ver=result.recipe_ver,
        model_ver=result.model_ver,
        calib_ver=result.calib_ver,
        cycle_time_ms=result.cycle_time_ms,
        timestamp_utc=result.timestamp_utc,
        measurements_json=[m.to_dict() for m in result.measurements],
        defects_json=[d.to_dict() for d in result.defects],
        anomaly_score=result.anomaly_score,
        image_refs_json=list(result.image_refs),
        traceability_json=serialize_traceability(result),
        defect_measurements_json=[m.to_dict() for m in result.defect_measurements],
        defect_count=len(result.defect_measurements),
        max_defect_area_um2=max(areas) if areas else None,
        # Largest defect first, per measure_defects()'s documented ordering.
        defect_type=result.defect_measurements[0].morphology
        if result.defect_measurements
        else None,
        drift_status=result.drift_status,
    )


def _from_record(record: InspectionRecord) -> InspectionResult:
    """Map an ORM record back to a domain :class:`~adaptivevision.common.InspectionResult`."""
    return InspectionResult(
        inspection_id=record.inspection_id,
        part_id=record.part_id,
        station_id=record.station_id,
        verdict=Verdict(record.verdict),
        recipe_ver=record.recipe_ver,
        model_ver=record.model_ver,
        calib_ver=record.calib_ver,
        cycle_time_ms=record.cycle_time_ms,
        timestamp_utc=_as_utc(record.timestamp_utc),
        measurements=tuple(Measurement.from_dict(m) for m in record.measurements_json),
        defects=tuple(Defect.from_dict(d) for d in record.defects_json),
        anomaly_score=record.anomaly_score,
        image_refs=tuple(record.image_refs_json),
        defect_measurements=tuple(
            DefectMeasurement.from_dict(m) for m in record.defect_measurements_json
        ),
        drift_status=record.drift_status,
    )


def _as_utc(value: datetime) -> datetime:
    """Normalize a stored timestamp to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class SqliteAdvisoryRepository(AdvisoryRepository):
    """An :class:`~adaptivevision.common.AdvisoryRepository` backed by the local SQLite database.

    Args:
        session_factory: The session factory to use for database access.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Initialize the repository."""
        self._session_factory = session_factory

    def save_report(
        self,
        inspection_id: str,
        evidence: InspectionEvidence,
        report: AdvisoryReport,
    ) -> None:
        """Persist an advisory report linked to ``inspection_id``.

        Raises:
            AdaptiveVisionError: On storage failure.
        """
        record = AdvisoryRecord(
            inspection_id=inspection_id,
            evidence_json=evidence.to_dict(),
            report_json=report.to_dict(),
            created_at=datetime.now(UTC),
        )
        try:
            with self._session_factory() as session:
                existing = session.scalar(
                    select(AdvisoryRecord).where(AdvisoryRecord.inspection_id == inspection_id)
                )
                if existing is not None:
                    session.delete(existing)
                    session.flush()
                session.add(record)
                session.commit()
        except Exception as exc:
            msg = f"Failed to persist advisory report for {inspection_id!r}: {exc}"
            raise AdaptiveVisionError(msg) from exc

    def get_report(self, inspection_id: str) -> AdvisoryReport | None:
        """Return the advisory report for ``inspection_id``, or ``None`` if absent."""
        try:
            with self._session_factory() as session:
                record = session.scalar(
                    select(AdvisoryRecord).where(AdvisoryRecord.inspection_id == inspection_id)
                )
        except Exception as exc:
            msg = f"Failed to query advisory report for {inspection_id!r}: {exc}"
            raise AdaptiveVisionError(msg) from exc
        return AdvisoryReport.from_dict(record.report_json) if record is not None else None


# =============================================================================
# Bounded local image archival
#
# A simple, bounded image store: archives raw image bytes to a local
# directory and returns a stable reference recorded in the inspection
# result's image_refs. Writes each image to a file named by its frame id and
# enforces a maximum number of retained images by evicting the oldest files.
# =============================================================================


class ImageStoreError(AdaptiveVisionError):
    """Failure to archive or retrieve an image."""


class LocalImageStore:
    """A bounded, local, on-disk image archive.

    Args:
        directory: Directory to store image files in. Created if missing.
        max_images: Maximum number of images to retain. When exceeded, the
            oldest files are evicted.
    """

    def __init__(self, directory: str | Path, *, max_images: int = 1000) -> None:
        """Initialize the store."""
        self._directory = Path(directory)
        self._max_images = max_images
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """Return the archive directory."""
        return self._directory

    def archive(self, frame_id: str, data: bytes) -> str:
        """Archive image ``data`` under ``frame_id``.

        Args:
            frame_id: Unique identifier of the frame (used as the file name).
            data: Raw image bytes to persist.

        Returns:
            A stable reference to the archived image.

        Raises:
            ImageStoreError: If the image cannot be written.
        """
        if not frame_id:
            msg = "frame_id must not be empty"
            raise ImageStoreError(msg)
        path = self._directory / f"{frame_id}.bin"
        try:
            path.write_bytes(data)
        except OSError as exc:
            msg = f"Failed to archive image {frame_id!r}: {exc}"
            raise ImageStoreError(msg) from exc
        self._evict_oldest()
        return str(path)

    def resolve(self, reference: str) -> Path:
        """Resolve an image reference to its on-disk path.

        Args:
            reference: The reference returned by :meth:`archive`.

        Returns:
            The :class:`~pathlib.Path` of the archived image.

        Raises:
            ImageStoreError: If the referenced image is missing.
        """
        path = Path(reference)
        if not path.exists():
            msg = f"Archived image not found: {reference!r}"
            raise ImageStoreError(msg)
        return path

    def _evict_oldest(self) -> None:
        """Evict the oldest files when the store exceeds its bound."""
        files = sorted(self._directory.glob("*.bin"), key=lambda p: p.stat().st_mtime)
        while len(files) > self._max_images:
            oldest = files.pop(0)
            with contextlib.suppress(OSError):
                oldest.unlink()


# =============================================================================
# Critical-path integration
#
# Persistence must stay off the inspection critical path: a database failure
# must never crash or block the inspection loop. This result handler is
# wired into the scheduler's on_result callback, persists each completed
# inspection result, and logs any failure clearly, preserving the
# inspection_id so an operator can trace a result from the logs to the
# database.
# =============================================================================


class PersistenceHandler:
    """Persists inspection results off the critical path.

    Args:
        repository: The result repository to persist to.
        image_store: Optional image store used to archive image references.
    """

    def __init__(
        self,
        repository: SqliteResultRepository,
        image_store: LocalImageStore | None = None,
    ) -> None:
        """Initialize the handler."""
        self._repository = repository
        self._image_store = image_store

    def on_result(self, result: InspectionResult) -> None:
        """Persist a completed inspection result.

        Failures are logged and swallowed so the inspection loop is never
        blocked or crashed by a persistence problem.

        Args:
            result: The completed inspection result.
        """
        try:
            self._repository.save_result(result)
        except Exception as exc:
            logger.error(
                "Failed to persist inspection result",
                extra={
                    "inspection_id": result.inspection_id,
                    "part_id": result.part_id,
                    "error": str(exc),
                },
            )
            return
        logger.info(
            "Inspection result persisted",
            extra={
                "inspection_id": result.inspection_id,
                "part_id": result.part_id,
                "verdict": result.verdict.value,
            },
        )


def make_persistence_handler(
    repository: SqliteResultRepository,
    image_store: LocalImageStore | None = None,
) -> Callable[[InspectionResult], None]:
    """Build an ``on_result`` callback that persists results.

    Args:
        repository: The result repository to persist to.
        image_store: Optional image store used to archive image references.

    Returns:
        A callable suitable for the scheduler's ``on_result`` hook.
    """
    handler = PersistenceHandler(repository, image_store)
    return handler.on_result


# =============================================================================
# Traceability and MES event export
#
# Traceability preserves the full lineage of an inspection so an operator can
# reconstruct exactly what happened for a given part. build_mes_payload()
# shapes the same data as a Manufacturing Execution System integration event
# -- there is no single universal MES schema (real integrations vary per
# vendor/site), so this is a reasonable, documented default shape, not a
# specific vendor integration.
# =============================================================================


def build_traceability_record(result: InspectionResult) -> dict[str, Any]:
    """Build a JSON-friendly traceability record for an inspection result.

    Args:
        result: The completed inspection result.

    Returns:
        A dictionary capturing the full inspection lineage.
    """
    return {
        "inspection_id": result.inspection_id,
        "part_id": result.part_id,
        "station_id": result.station_id,
        "recipe_version": result.recipe_ver,
        "model_version": result.model_ver,
        "calibration_version": result.calib_ver,
        "verdict": result.verdict.value,
        "cycle_time_ms": result.cycle_time_ms,
        "start_timestamp": _iso(result.timestamp_utc),
        "end_timestamp": _iso(result.timestamp_utc),
        "defects": [d.to_dict() for d in result.defects],
        "measurements": [m.to_dict() for m in result.measurements],
        "anomaly_score": result.anomaly_score,
        "image_refs": list(result.image_refs),
    }


def serialize_traceability(result: InspectionResult) -> str:
    """Serialize a traceability record to a JSON string.

    Args:
        result: The completed inspection result.

    Returns:
        A JSON string capturing the inspection lineage.
    """
    return json.dumps(build_traceability_record(result), sort_keys=True)


def build_mes_payload(result: InspectionResult) -> dict[str, Any]:
    """Build a JSON MES (Manufacturing Execution System) event payload for a result (Milestone M21).

    There is no single universal MES event schema -- real integrations vary
    per vendor and site (SEMI E10/E30-style equipment events, proprietary
    line-controller formats, etc.). This is a reasonable, documented default
    shape: the fields a downstream MES/SCADA system typically keys routing or
    alarm logic off (disposition, defect summary, drift status) promoted to
    the top level, with the full traceability record nested under
    ``detail`` for audit trails -- not a specific vendor integration.

    Args:
        result: The completed inspection result.

    Returns:
        A dictionary shaped as one MES inspection event.
    """
    areas = [m.area_um2 for m in result.defect_measurements]
    return {
        "event_type": "INSPECTION_RESULT",
        "event_timestamp": _iso(result.timestamp_utc),
        "station_id": result.station_id,
        "part_id": result.part_id,
        "inspection_id": result.inspection_id,
        "disposition": result.verdict.value.upper(),
        "defect_summary": {
            "defect_count": len(result.defect_measurements),
            "max_defect_area_um2": max(areas) if areas else None,
            "dominant_defect_type": (
                result.defect_measurements[0].morphology if result.defect_measurements else None
            ),
        },
        "drift_status": result.drift_status,
        "detail": build_traceability_record(result),
    }


def serialize_mes_payload(result: InspectionResult) -> str:
    """Serialize an MES event payload to a JSON string.

    Args:
        result: The completed inspection result.

    Returns:
        A JSON string of the MES event payload built by :func:`build_mes_payload`.
    """
    return json.dumps(build_mes_payload(result), sort_keys=True)


def _iso(value: datetime) -> str:
    """Return an ISO-8601 string for ``value``."""
    return value.isoformat()
