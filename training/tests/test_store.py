"""run_id hashing and resume/skip logic -- the narrow test surface this refactor asked for."""

from __future__ import annotations

import pytest
from training.store import (
    completed_run_ids,
    compute_run_id,
    finish_run,
    open_database,
    reset_incomplete,
    start_run,
    update_columns,
)


def _base_kwargs() -> dict:
    return {
        "method": "patchcore_resnet18",
        "regime": "oneclass",
        "config_key": "mvtec/bottle",
        "seed": 1,
    }


def test_compute_run_id_deterministic() -> None:
    a = compute_run_id(**_base_kwargs())
    b = compute_run_id(**_base_kwargs())
    assert a == b


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 2},
        {"method": "patchcore_wide_resnet50_2"},
        {"config_key": "mvtec/hazelnut"},
        {"regime": "multiclass"},
    ],
)
def test_compute_run_id_changes_with_identity_field(override: dict) -> None:
    base = compute_run_id(**_base_kwargs())
    changed = compute_run_id(**{**_base_kwargs(), **override})
    assert base != changed


def test_compute_run_id_changes_with_defect_kind() -> None:
    combined = compute_run_id(**_base_kwargs(), defect_kind=None)
    structural = compute_run_id(**_base_kwargs(), defect_kind="structural")
    logical = compute_run_id(**_base_kwargs(), defect_kind="logical")
    assert len({combined, structural, logical}) == 3


def test_compute_run_id_changes_with_severstal_prevalence() -> None:
    base = compute_run_id(**_base_kwargs(), severstal_target_prevalence=None)
    subsampled = compute_run_id(**_base_kwargs(), severstal_target_prevalence=0.275)
    assert base != subsampled


def test_compute_run_id_changes_with_schema_version() -> None:
    v1 = compute_run_id(**_base_kwargs(), schema_version=1)
    v2 = compute_run_id(**_base_kwargs(), schema_version=2)
    assert v1 != v2


def test_compute_run_id_stable_field_order() -> None:
    """dict insertion order must not matter -- json.dumps(sort_keys=True)."""
    a = compute_run_id(method="dfm_wide_resnet50_2", regime="oneclass", config_key="kolektor", seed=0)
    b = compute_run_id(seed=0, config_key="kolektor", regime="oneclass", method="dfm_wide_resnet50_2")
    assert a == b


def _identity(**overrides) -> dict:
    base = {
        "regime": "oneclass",
        "method": "patchcore_resnet18",
        "family": "patchcore",
        "backend": "native",
        "config": "mvtec/bottle",
        "dataset": "mvtec",
        "category": "bottle",
        "height": 256,
        "width": 256,
        "seed": 1,
    }
    base.update(overrides)
    return base


def test_open_database_creates_missing_parent_directory(tmp_path) -> None:
    """SQLite cannot create the file inside a directory that doesn't exist
    yet -- open_database must create it first, the way the old JSONL
    append_row used to."""
    nested_path = tmp_path / "does" / "not" / "exist" / "test.db"
    assert not nested_path.parent.exists()
    open_database(nested_path)  # must not raise
    assert nested_path.exists()


def test_start_then_finish_updates_same_row_not_duplicate(tmp_path) -> None:
    _, session_factory = open_database(tmp_path / "test.db")
    run_id = compute_run_id(**_base_kwargs())
    start_run(session_factory, run_id, _identity())
    finish_run(session_factory, run_id, {"status": "ok", "auroc": 0.99})

    from sqlalchemy import select
    from training.store import RunRow, session_scope

    with session_scope(session_factory) as session:
        rows = list(session.scalars(select(RunRow).where(RunRow.run_id == run_id)))
    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].auroc == pytest.approx(0.99)


def test_update_columns_backfills_existing_row(tmp_path) -> None:
    _, session_factory = open_database(tmp_path / "test.db")
    run_id = compute_run_id(**_base_kwargs())
    start_run(session_factory, run_id, _identity())
    finish_run(session_factory, run_id, {"status": "ok", "auroc": 0.9})

    found = update_columns(session_factory, run_id, {"inference_latency_ms_p50": 7.2})
    assert found is True

    from sqlalchemy import select
    from training.store import RunRow, session_scope

    with session_scope(session_factory) as session:
        row = session.scalar(select(RunRow).where(RunRow.run_id == run_id))
    assert row.inference_latency_ms_p50 == pytest.approx(7.2)
    assert row.auroc == pytest.approx(0.9)  # untouched by the backfill


def test_update_columns_missing_row_returns_false(tmp_path) -> None:
    _, session_factory = open_database(tmp_path / "test.db")
    found = update_columns(session_factory, "does-not-exist", {"auroc": 0.5})
    assert found is False


def test_reset_incomplete_deletes_running_and_failed_keeps_ok(tmp_path) -> None:
    _, session_factory = open_database(tmp_path / "test.db")
    ok_id = compute_run_id(method="a", regime="oneclass", config_key="c1", seed=1)
    running_id = compute_run_id(method="b", regime="oneclass", config_key="c1", seed=1)
    failed_id = compute_run_id(method="c", regime="oneclass", config_key="c1", seed=1)

    start_run(session_factory, ok_id, _identity(method="a"))
    finish_run(session_factory, ok_id, {"status": "ok"})
    start_run(session_factory, running_id, _identity(method="b"))
    # left "running" -- simulates a crash before finish_run
    start_run(session_factory, failed_id, _identity(method="c"))
    finish_run(session_factory, failed_id, {"status": "failed", "error": "boom"})

    scope = {ok_id, running_id, failed_id}
    deleted = reset_incomplete(session_factory, scope)
    assert deleted == 2
    assert completed_run_ids(session_factory, scope) == {ok_id}


def test_reset_incomplete_scoped_not_table_wide(tmp_path) -> None:
    """A reset for one plan must never touch another plan's in-flight rows."""
    _, session_factory = open_database(tmp_path / "test.db")
    other_running_id = compute_run_id(method="other", regime="oneclass", config_key="c1", seed=1)
    start_run(session_factory, other_running_id, _identity(method="other"))

    deleted = reset_incomplete(session_factory, {"some-unrelated-run-id"})
    assert deleted == 0
    # The row outside scope must still exist.
    from sqlalchemy import select
    from training.store import RunRow, session_scope

    with session_scope(session_factory) as session:
        assert session.scalar(select(RunRow).where(RunRow.run_id == other_running_id)) is not None


def test_reset_incomplete_force_also_deletes_ok(tmp_path) -> None:
    _, session_factory = open_database(tmp_path / "test.db")
    ok_id = compute_run_id(method="a", regime="oneclass", config_key="c1", seed=1)
    start_run(session_factory, ok_id, _identity(method="a"))
    finish_run(session_factory, ok_id, {"status": "ok"})

    deleted = reset_incomplete(session_factory, {ok_id}, force=True)
    assert deleted == 1
    assert completed_run_ids(session_factory, {ok_id}) == set()


def test_completed_run_ids_only_counts_ok(tmp_path) -> None:
    _, session_factory = open_database(tmp_path / "test.db")
    ok_id = compute_run_id(method="a", regime="oneclass", config_key="c1", seed=1)
    running_id = compute_run_id(method="b", regime="oneclass", config_key="c1", seed=1)
    start_run(session_factory, ok_id, _identity(method="a"))
    finish_run(session_factory, ok_id, {"status": "ok"})
    start_run(session_factory, running_id, _identity(method="b"))

    assert completed_run_ids(session_factory, {ok_id, running_id}) == {ok_id}


def test_start_run_upserts_over_an_existing_ok_row(tmp_path) -> None:
    """A multiclass job retried whole (because a sibling category crashed)
    re-registers every category's run_id, including ones already "ok" --
    start_run must not hit the run_id UNIQUE constraint for those."""
    _, session_factory = open_database(tmp_path / "test.db")
    run_id = compute_run_id(**_base_kwargs())
    start_run(session_factory, run_id, _identity())
    finish_run(session_factory, run_id, {"status": "ok", "auroc": 0.9})

    # Re-registering the same run_id (simulating a whole-family retry) must
    # not raise, and must reset the row to "running".
    start_run(session_factory, run_id, _identity())

    from sqlalchemy import select
    from training.store import RunRow, session_scope

    with session_scope(session_factory) as session:
        rows = list(session.scalars(select(RunRow).where(RunRow.run_id == run_id)))
    assert len(rows) == 1
    assert rows[0].status == "running"


def test_full_plan_resume_simulated_crash(tmp_path) -> None:
    """5 synthetic runs: 1-2 finish clean, 3 dies mid-flight, 4-5 never start.

    A fresh reset+lookup pass (what a relaunch does) must: keep 1-2 as done,
    delete 3 so it reappears pending, and leave 4-5 pending untouched. No
    GPU/model work -- this exercises store.py's whole crash/resume contract
    against an in-memory-backed file DB.
    """
    _, session_factory = open_database(tmp_path / "test.db")

    ids = [
        compute_run_id(method=f"m{i}", regime="oneclass", config_key="mvtec/bottle", seed=1)
        for i in range(1, 6)
    ]
    id1, id2, id3, id4, id5 = ids

    for run_id, method in [(id1, "m1"), (id2, "m2")]:
        start_run(session_factory, run_id, _identity(method=method))
        finish_run(session_factory, run_id, {"status": "ok", "auroc": 0.9})

    # Spec 3: started, then the process died before finish_run ran.
    start_run(session_factory, id3, _identity(method="m3"))

    # Specs 4-5 (id4, id5): never touched at all.

    scope = set(ids)
    deleted = reset_incomplete(session_factory, scope)
    assert deleted == 1  # only the crashed "running" row

    done = completed_run_ids(session_factory, scope)
    assert done == {id1, id2}

    pending = scope - done
    assert pending == {id3, id4, id5}
