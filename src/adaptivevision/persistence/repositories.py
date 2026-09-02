"""Persistence repositories (Milestone M4; M19 adds advisory reports).

This module implements the existing
:class:`~adaptivevision.common.interfaces.ResultRepository` seam against the
local SQLite database. It maps :class:`~adaptivevision.common.result.InspectionResult`
domain objects to the ORM :class:`~adaptivevision.persistence.models_orm.InspectionRecord`
at the repository boundary, keeping the domain layer free of SQLAlchemy types.

Storage failures surface as the base
:class:`~adaptivevision.common.errors.AdaptiveVisionError`, matching the
contract declared on the seam.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from adaptivevision.common.enums import Verdict
from adaptivevision.common.errors import AdaptiveVisionError
from adaptivevision.common.interfaces import AdvisoryRepository, ResultRepository
from adaptivevision.common.result import (
    AdvisoryReport,
    Defect,
    DefectMeasurement,
    InspectionEvidence,
    InspectionResult,
)
from adaptivevision.common.types import Measurement
from adaptivevision.persistence.models_orm import AdvisoryRecord, InspectionRecord
from adaptivevision.persistence.traceability import serialize_traceability


class SqliteResultRepository(ResultRepository):
    """A :class:`ResultRepository` backed by the local SQLite database.

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
            The matching :class:`InspectionResult`, or ``None``.
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
            A tuple of :class:`InspectionResult`, most-recent first.
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
    """Map a domain :class:`InspectionResult` to an ORM record."""
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
        defect_type=result.defect_measurements[0].morphology if result.defect_measurements else None,
        drift_status=result.drift_status,
    )


def _from_record(record: InspectionRecord) -> InspectionResult:
    """Map an ORM record back to a domain :class:`InspectionResult`."""
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
    """An :class:`AdvisoryRepository` backed by the local SQLite database.

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
