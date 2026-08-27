"""Integration tests for the M4 persistence layer.

These tests exercise the persistence package end-to-end against an in-memory
SQLite database: the database layer, the result repository, traceability, the
image store, and the persistence handler that is wired into the station's
``on_result`` hook.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from adaptivevision.common.enums import DefectClass, Severity, Verdict
from adaptivevision.common.errors import AdaptiveVisionError
from adaptivevision.common.result import (
    AdvisoryReport,
    Defect,
    InspectionEvidence,
    InspectionResult,
)
from adaptivevision.common.types import ROI, Measurement
from adaptivevision.persistence.database import (
    build_engine,
    init_db,
    make_session_factory,
    open_database,
    session_scope,
)
from adaptivevision.persistence.image_store import ImageStoreError, LocalImageStore
from adaptivevision.persistence.integration import (
    PersistenceHandler,
    make_persistence_handler,
)
from adaptivevision.persistence.models_orm import InspectionRecord
from adaptivevision.persistence.repositories import (
    SqliteAdvisoryRepository,
    SqliteResultRepository,
)
from adaptivevision.persistence.traceability import (
    build_traceability_record,
    serialize_traceability,
)


def _result(
    inspection_id: str = "insp-1",
    part_id: str = "part-1",
    verdict: Verdict = Verdict.PASS,
    timestamp_utc: datetime = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
) -> InspectionResult:
    return InspectionResult(
        inspection_id=inspection_id,
        part_id=part_id,
        station_id="station-A",
        verdict=verdict,
        recipe_ver="recipe-3",
        model_ver="model-2",
        calib_ver="calib-1",
        cycle_time_ms=142.5,
        timestamp_utc=timestamp_utc,
        measurements=(
            Measurement(name="width", value=10.1, unit="mm"),
            Measurement(name="height", value=5.2, unit="mm"),
        ),
        defects=(
            Defect(
                defect_class=DefectClass.SCRATCH,
                severity=Severity.MAJOR,
                score=0.87,
                roi=ROI(label="r", x=0.0, y=0.0, width=1.0, height=1.0),
                description="hairline scratch",
            ),
        ),
        anomaly_score=0.91,
        image_refs=("img/raw-1.png", "img/overlay-1.png"),
    )


@pytest.fixture()
def session_factory():
    """Provide an in-memory database session factory."""
    engine = build_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture()
def repository(session_factory) -> SqliteResultRepository:
    """Provide a repository bound to an in-memory database."""
    return SqliteResultRepository(session_factory)


@pytest.fixture()
def advisory_repository(session_factory) -> SqliteAdvisoryRepository:
    """Provide an advisory repository bound to an in-memory database."""
    return SqliteAdvisoryRepository(session_factory)


def _evidence(sample_id: str = "insp-1") -> InspectionEvidence:
    return InspectionEvidence(
        sample_id=sample_id,
        category="bottle",
        anomaly_score=0.9,
        severity=Severity.MAJOR,
        model_ver="patchcore-v1",
    )


def _report(*, severity: Severity = Severity.MAJOR) -> AdvisoryReport:
    return AdvisoryReport(
        defect_classification="crack",
        severity=severity,
        confidence_score=0.7,
        root_cause_hypothesis="Likely mold defect.",
        recommended_actions=("inspect mold",),
    )


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------


def test_open_database_in_memory_initializes_schema() -> None:
    engine, factory = open_database(":memory:")
    assert engine is not None
    with factory() as session:
        assert session.execute(InspectionRecord.__table__.select().limit(1)).first() is None


def test_init_db_is_idempotent() -> None:
    engine = build_engine("sqlite:///:memory:")
    init_db(engine)
    init_db(engine)  # must not raise
    assert engine is not None


def test_session_scope_commits_on_success(session_factory) -> None:
    with session_scope(session_factory) as session:
        session.add(
            InspectionRecord(
                inspection_id="insp-commit",
                part_id="p",
                station_id="s",
                verdict="pass",
                recipe_ver="r",
                model_ver="m",
                calib_ver="c",
                cycle_time_ms=1.0,
                timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC),
            )
        )
    with session_factory() as session:
        assert session.get(InspectionRecord, 1) is not None


def test_session_scope_rolls_back_on_error(session_factory) -> None:
    with pytest.raises(RuntimeError), session_scope(session_factory) as session:
        session.add(
            InspectionRecord(
                inspection_id="insp-rollback",
                part_id="p",
                station_id="s",
                verdict="pass",
                recipe_ver="r",
                model_ver="m",
                calib_ver="c",
                cycle_time_ms=1.0,
                timestamp_utc=datetime(2026, 1, 2, tzinfo=UTC),
            )
        )
        raise RuntimeError("boom")
    with session_factory() as session:
        assert session.get(InspectionRecord, 1) is None


# ---------------------------------------------------------------------------
# Result repository
# ---------------------------------------------------------------------------


def test_repository_save_and_get_roundtrip(repository: SqliteResultRepository) -> None:
    original = _result()
    repository.save_result(original)
    restored = repository.get_result("insp-1")
    assert restored is not None
    assert restored == original


def test_repository_get_missing_returns_none(
    repository: SqliteResultRepository,
) -> None:
    assert repository.get_result("does-not-exist") is None


def test_repository_list_orders_most_recent_first(
    repository: SqliteResultRepository,
) -> None:
    older = _result(
        inspection_id="insp-old",
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = _result(
        inspection_id="insp-new",
        timestamp_utc=datetime(2026, 1, 3, tzinfo=UTC),
    )
    repository.save_result(older)
    repository.save_result(newer)
    results = repository.list_results()
    assert [r.inspection_id for r in results] == ["insp-new", "insp-old"]


def test_repository_list_respects_limit_and_offset(
    repository: SqliteResultRepository,
) -> None:
    for i in range(5):
        repository.save_result(
            _result(
                inspection_id=f"insp-{i}",
                timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
    page = repository.list_results(limit=2, offset=1)
    assert len(page) == 2


def test_repository_preserves_verdict_and_lineage(
    repository: SqliteResultRepository,
) -> None:
    original = _result(verdict=Verdict.FAIL)
    repository.save_result(original)
    restored = repository.get_result("insp-1")
    assert restored is not None
    assert restored.verdict == Verdict.FAIL
    assert restored.recipe_ver == "recipe-3"
    assert restored.model_ver == "model-2"
    assert restored.calib_ver == "calib-1"


def test_repository_save_duplicate_inspection_id_raises(
    repository: SqliteResultRepository,
) -> None:
    repository.save_result(_result())
    with pytest.raises(AdaptiveVisionError):
        repository.save_result(_result())


def test_repository_get_wraps_storage_failure() -> None:
    repository = SqliteResultRepository(_broken_session_factory())
    with pytest.raises(AdaptiveVisionError):
        repository.get_result("insp-1")


def test_repository_list_wraps_storage_failure() -> None:
    repository = SqliteResultRepository(_broken_session_factory())
    with pytest.raises(AdaptiveVisionError):
        repository.list_results()


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


def test_build_traceability_record_captures_lineage() -> None:
    record = build_traceability_record(_result())
    assert record["inspection_id"] == "insp-1"
    assert record["part_id"] == "part-1"
    assert record["station_id"] == "station-A"
    assert record["recipe_version"] == "recipe-3"
    assert record["model_version"] == "model-2"
    assert record["calibration_version"] == "calib-1"
    assert record["verdict"] == "pass"
    assert record["cycle_time_ms"] == 142.5
    assert record["anomaly_score"] == 0.91
    assert record["image_refs"] == ["img/raw-1.png", "img/overlay-1.png"]
    assert len(record["defects"]) == 1
    assert len(record["measurements"]) == 2


def test_serialize_traceability_is_valid_json() -> None:
    payload = serialize_traceability(_result())
    parsed = json.loads(payload)
    assert parsed["inspection_id"] == "insp-1"


# ---------------------------------------------------------------------------
# Image store
# ---------------------------------------------------------------------------


def test_image_store_archive_and_resolve(tmp_path) -> None:
    store = LocalImageStore(tmp_path / "images")
    reference = store.archive("frame-1", b"\x00\x01\x02")
    assert store.resolve(reference).read_bytes() == b"\x00\x01\x02"


def test_image_store_rejects_empty_frame_id(tmp_path) -> None:
    store = LocalImageStore(tmp_path / "images")
    with pytest.raises(ImageStoreError):
        store.archive("", b"data")


def test_image_store_resolve_missing_raises(tmp_path) -> None:
    store = LocalImageStore(tmp_path / "images")
    with pytest.raises(ImageStoreError):
        store.resolve(str(tmp_path / "images" / "missing.bin"))


def test_image_store_evicts_oldest_when_bounded(tmp_path) -> None:
    store = LocalImageStore(tmp_path / "images", max_images=2)
    store.archive("a", b"1")
    store.archive("b", b"2")
    store.archive("c", b"3")
    remaining = sorted(p.name for p in (tmp_path / "images").glob("*.bin"))
    assert remaining == ["b.bin", "c.bin"]


# ---------------------------------------------------------------------------
# Persistence handler (off the critical path)
# ---------------------------------------------------------------------------


def test_persistence_handler_persists_result(
    repository: SqliteResultRepository,
) -> None:
    handler = PersistenceHandler(repository)
    handler.on_result(_result())
    assert repository.get_result("insp-1") is not None


def test_make_persistence_handler_returns_callable(
    repository: SqliteResultRepository,
) -> None:
    handler = make_persistence_handler(repository)
    handler(_result())
    assert repository.get_result("insp-1") is not None


def test_persistence_handler_swallows_failures(
    repository: SqliteResultRepository,
) -> None:
    # A duplicate inspection id forces a storage failure; the handler must log
    # and swallow it rather than raise, keeping the inspection loop alive.
    handler = PersistenceHandler(repository)
    handler.on_result(_result())
    handler.on_result(_result())  # duplicate -> failure, must not raise


# ---------------------------------------------------------------------------
# Advisory repository (Milestone M19)
# ---------------------------------------------------------------------------


def test_advisory_repository_save_and_get_roundtrip(
    advisory_repository: SqliteAdvisoryRepository,
) -> None:
    report = _report()
    advisory_repository.save_report("insp-1", _evidence(), report)
    restored = advisory_repository.get_report("insp-1")
    assert restored == report


def test_advisory_repository_get_missing_returns_none(
    advisory_repository: SqliteAdvisoryRepository,
) -> None:
    assert advisory_repository.get_report("does-not-exist") is None


def test_advisory_repository_save_overwrites_existing_report(
    advisory_repository: SqliteAdvisoryRepository,
) -> None:
    advisory_repository.save_report("insp-1", _evidence(), _report(severity=Severity.MAJOR))
    advisory_repository.save_report("insp-1", _evidence(), _report(severity=Severity.CRITICAL))
    restored = advisory_repository.get_report("insp-1")
    assert restored is not None
    assert restored.severity == Severity.CRITICAL


def test_advisory_repository_is_independent_of_result_repository(
    repository: SqliteResultRepository,
    advisory_repository: SqliteAdvisoryRepository,
) -> None:
    repository.save_result(_result())
    advisory_repository.save_report("insp-1", _evidence(), _report())
    assert repository.get_result("insp-1") is not None
    assert advisory_repository.get_report("insp-1") is not None


def _broken_session_factory():
    """A session factory that fails before yielding a session."""

    def _raise():
        msg = "simulated database failure"
        raise RuntimeError(msg)

    return _raise


def test_advisory_repository_save_wraps_storage_failure() -> None:
    repository = SqliteAdvisoryRepository(_broken_session_factory())
    with pytest.raises(AdaptiveVisionError):
        repository.save_report("insp-1", _evidence(), _report())


def test_advisory_repository_get_wraps_storage_failure() -> None:
    repository = SqliteAdvisoryRepository(_broken_session_factory())
    with pytest.raises(AdaptiveVisionError):
        repository.get_report("insp-1")


def test_as_utc_leaves_timezone_aware_timestamp_unchanged() -> None:
    from datetime import timedelta, timezone

    from adaptivevision.persistence.repositories import _as_utc

    aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=5)))
    result = _as_utc(aware)
    assert result.tzinfo == UTC
    assert result == aware
