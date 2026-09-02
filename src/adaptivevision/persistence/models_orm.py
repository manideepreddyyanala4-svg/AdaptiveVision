"""SQLAlchemy ORM models for the local edge database (Milestone M4; M19 adds advisory).

These models are the *persistence* representation of an inspection result. They
are deliberately kept completely separate from the M1 domain models in
:mod:`adaptivevision.common.result`: the domain objects never import or depend
on SQLAlchemy types, and the ORM models never leak into the domain layer.

The mapping between the two representations happens only at the repository
boundary (:mod:`adaptivevision.persistence.repositories`).

The schema is intentionally small at M4. Lineage fields (recipe / model /
calibration versions) are stored as first-class columns so they can be queried
directly; the richer, nested result data (measurements, defects, image
references) is serialized to JSON columns via the domain ``to_dict`` /
``from_dict`` contract.

Milestone M19 adds :class:`AdvisoryRecord` as a *second*, independent table -
it never modifies :class:`InspectionRecord`, matching the read-only relation
between an inspection and its (optional) advisory report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all persistence ORM models."""


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
            or ``"particle"`` -- see
            :mod:`adaptivevision.inspection.anomaly.metrology`), or ``None``
            when no defect was measured.
        drift_status: Sensor/illumination drift status in effect at the time
            of this inspection, or ``None`` when no drift detector was wired
            in (see :mod:`adaptivevision.monitoring.drift`).
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
        evidence_json: Serialized :class:`~adaptivevision.common.result.InspectionEvidence`.
        report_json: Serialized :class:`~adaptivevision.common.result.AdvisoryReport`.
        created_at: Time the report was persisted (timezone-aware, UTC).
    """

    __tablename__ = "advisory_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
