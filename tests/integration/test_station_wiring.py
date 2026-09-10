"""Integration tests for the M22 station wiring: image archive, PLC reject
dispatch, and the off-thread advisory worker."""

from __future__ import annotations

import importlib.util
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adaptivevision.api import create_app
from adaptivevision.app import (
    build_advisory_worker,
    build_image_store,
    build_manual_inspector,
    build_reject_dispatcher,
    chain_handlers,
)
from adaptivevision.common import (
    DEFAULT_LOT_ID,
    AdvisoryReport,
    CommsError,
    Defect,
    DefectClass,
    DefectMeasurement,
    InspectionEvidence,
    InspectionResult,
    ResultRepository,
    Severity,
    Verdict,
)
from adaptivevision.communication import (
    ModbusTcpTransport,
    RejectDispatcher,
    SocketModbusClient,
)
from adaptivevision.config import StationConfig
from adaptivevision.explanation import AdvisoryWorker
from adaptivevision.orchestration import InspectionPipeline, ManualInspectionService
from adaptivevision.storage import (
    ImageStoreError,
    LocalImageStore,
    SqliteAdvisoryRepository,
    open_database,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(**extra: object) -> StationConfig:
    return StationConfig(station_id="s1", log_level="INFO", extra=dict(extra))


def _result(
    verdict: Verdict = Verdict.FAIL,
    inspection_id: str = "insp-1",
    **overrides: Any,
) -> InspectionResult:
    fields: dict[str, Any] = {
        "inspection_id": inspection_id,
        "part_id": "part-1",
        "station_id": "s1",
        "verdict": verdict,
        "recipe_ver": "demo-bottle",
        "model_ver": "patchcore",
        "calib_ver": "",
        "cycle_time_ms": 715.0,
        "timestamp_utc": datetime.now(UTC),
        "anomaly_score": 1.0,
        "defects": (Defect(defect_class=DefectClass.ANOMALY, severity=Severity.MAJOR),),
        "defect_measurements": (
            DefectMeasurement(
                bbox=(12, 30, 40, 9),
                area_px2=210,
                area_um2=1312.5,
                aspect_ratio=4.44,
                morphology="scratch",
            ),
        ),
    }
    fields.update(overrides)
    return InspectionResult(**fields)


# -----------------------------------------------------------------------------
# Image archive
# -----------------------------------------------------------------------------


def test_archive_image_writes_a_decodable_png(tmp_path: Path) -> None:
    store = LocalImageStore(tmp_path)
    image = np.full((8, 6, 3), (10, 20, 30), dtype=np.uint8)

    reference = store.archive_image("frame-1", image)

    data = Path(reference).read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    import cv2

    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (8, 6, 3)
    # Stored RGB, decoded by cv2 as BGR: the channel order must round-trip.
    assert tuple(decoded[0, 0]) == (30, 20, 10)


def test_archive_image_scales_float_frames_to_8bit(tmp_path: Path) -> None:
    """A rectified/preprocessed frame may be float; the archive must still
    produce a viewable image rather than clipping everything to black."""
    store = LocalImageStore(tmp_path)
    image = np.linspace(0.0, 4.0, 16, dtype=np.float32).reshape(4, 4)

    reference = store.archive_image("frame-float", image)

    import cv2

    decoded = cv2.imdecode(
        np.frombuffer(Path(reference).read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    assert decoded.max() == 255
    assert decoded.dtype == np.uint8


def test_path_for_resolves_an_archived_id(tmp_path: Path) -> None:
    store = LocalImageStore(tmp_path)
    store.archive_image("frame-9", np.zeros((2, 2), dtype=np.uint8))
    assert store.path_for("frame-9").exists()


@pytest.mark.parametrize("image_id", ["../secrets", "a/b", "", ".", ".."])
def test_path_for_rejects_traversal(tmp_path: Path, image_id: str) -> None:
    """An id is a single archive entry, never a path -- a request must not be
    able to read outside the archive directory."""
    store = LocalImageStore(tmp_path)
    with pytest.raises(ImageStoreError):
        store.path_for(image_id)


def test_path_for_missing_image_raises(tmp_path: Path) -> None:
    with pytest.raises(ImageStoreError):
        LocalImageStore(tmp_path).path_for("never-archived")


def test_image_store_evicts_oldest_beyond_bound(tmp_path: Path) -> None:
    store = LocalImageStore(tmp_path, max_images=2)
    for index in range(4):
        store.archive_image(f"frame-{index}", np.zeros((2, 2), dtype=np.uint8))
    assert len(list(tmp_path.glob("*.bin"))) == 2


# -----------------------------------------------------------------------------
# Image API route
# -----------------------------------------------------------------------------


class _EmptyRepository(ResultRepository):
    def save_result(self, result: InspectionResult) -> None:
        """Unused."""

    def get_result(self, inspection_id: str) -> InspectionResult | None:
        return None

    def list_results(self, *, limit: int = 100, offset: int = 0) -> tuple[InspectionResult, ...]:
        return ()


def test_image_route_serves_the_archived_png(tmp_path: Path) -> None:
    store = LocalImageStore(tmp_path)
    store.archive_image("frame-7", np.full((4, 4, 3), 200, dtype=np.uint8))

    with TestClient(create_app(_EmptyRepository(), image_store=store)) as client:
        response = client.get("/api/v1/images/frame-7")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_image_route_404s_for_unknown_id(tmp_path: Path) -> None:
    with TestClient(create_app(_EmptyRepository(), image_store=LocalImageStore(tmp_path))) as c:
        assert c.get("/api/v1/images/nope").status_code == 404


def test_image_routes_absent_without_a_store() -> None:
    """The archive is optional: with none configured the routes must not exist."""
    with TestClient(create_app(_EmptyRepository())) as client:
        assert client.get("/api/v1/images/frame-7").status_code == 404


# -----------------------------------------------------------------------------
# PLC reject dispatch
# -----------------------------------------------------------------------------


class _FakeTransport:
    """Records coil writes; optionally fails like an unreachable PLC."""

    def __init__(self, *, fail: bool = False) -> None:
        self.writes: list[tuple[int, bool]] = []
        self._fail = fail

    def write_coil(self, address: int, value: bool) -> None:
        if self._fail:
            raise CommsError("PLC unreachable")
        self.writes.append((address, value))


def test_reject_dispatcher_asserts_coil_on_fail() -> None:
    transport = _FakeTransport()
    RejectDispatcher(transport, coil_address=3).on_result(_result(Verdict.FAIL))  # type: ignore[arg-type]
    assert transport.writes == [(3, True)]


def test_reject_dispatcher_clears_coil_on_pass() -> None:
    transport = _FakeTransport()
    RejectDispatcher(transport, coil_address=3).on_result(_result(Verdict.PASS))  # type: ignore[arg-type]
    assert transport.writes == [(3, False)]


def test_reject_dispatcher_routes_review_to_a_human_not_the_ejector() -> None:
    transport = _FakeTransport()
    RejectDispatcher(transport).on_result(_result(Verdict.REVIEW))  # type: ignore[arg-type]
    assert transport.writes == [(3, False)]


def test_reject_dispatcher_survives_a_dead_plc() -> None:
    """A PLC outage is an operations problem, not a reason to stop inspecting."""
    transport = _FakeTransport(fail=True)
    RejectDispatcher(transport).on_result(_result(Verdict.FAIL))  # type: ignore[arg-type]
    assert transport.writes == []


# -----------------------------------------------------------------------------
# Modbus client against the bench simulator
# -----------------------------------------------------------------------------


def _load_mock_plc() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mock_plc", PROJECT_ROOT / "scripts" / "mock_plc.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plc_server() -> Any:
    """Run the bench PLC simulator on an ephemeral port for one test."""
    module = _load_mock_plc()
    server = module.MockPlcServer("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_socket_client_round_trips_coils_and_registers(plc_server: Any) -> None:
    """The station's own client must speak to the simulator end-to-end."""
    port = plc_server.server_address[1]
    transport = ModbusTcpTransport(SocketModbusClient("127.0.0.1", port))
    transport.connect()
    try:
        assert transport.is_connected() is True

        transport.write_coil(3, True)
        assert transport.read_coils(0, 5) == (False, False, False, True, False)

        transport.write_coil(3, False)
        assert transport.read_coils(3, 1) == (False,)

        transport.write_registers(10, [1234, 42])
        assert transport.read_registers(10, 2) == (1234, 42)
    finally:
        transport.disconnect()


def test_reject_dispatcher_drives_the_simulator(plc_server: Any) -> None:
    port = plc_server.server_address[1]
    transport = ModbusTcpTransport(SocketModbusClient("127.0.0.1", port))
    transport.connect()
    try:
        dispatcher = RejectDispatcher(transport, coil_address=3)
        dispatcher.on_result(_result(Verdict.FAIL))
        assert transport.read_coils(3, 1) == (True,)
        dispatcher.on_result(_result(Verdict.PASS))
        assert transport.read_coils(3, 1) == (False,)
    finally:
        transport.disconnect()


def test_socket_client_raises_when_not_connected() -> None:
    transport = ModbusTcpTransport(SocketModbusClient("127.0.0.1", 1))
    with pytest.raises(CommsError):
        transport.write_coil(3, True)


# -----------------------------------------------------------------------------
# Advisory worker
# -----------------------------------------------------------------------------


class _FakeEngine:
    """Advisory engine that echoes the deterministic severity, as required."""

    def __init__(self) -> None:
        self.seen: list[InspectionEvidence] = []

    def generate_report(self, evidence: InspectionEvidence) -> AdvisoryReport:
        self.seen.append(evidence)
        return AdvisoryReport(
            defect_classification="scratch",
            severity=evidence.severity,
            confidence_score=0.9,
            root_cause_hypothesis="Tooling contact.",
            recommended_actions=("Check the mold.",),
            is_fallback=False,
        )


@pytest.fixture
def advisory_repo(tmp_path: Path) -> SqliteAdvisoryRepository:
    """A file-backed repository: the worker persists from another thread."""
    _, session_factory = open_database(tmp_path / "adv.db")
    return SqliteAdvisoryRepository(session_factory)


def test_worker_generates_and_persists_for_a_failing_part(
    advisory_repo: SqliteAdvisoryRepository,
) -> None:
    engine = _FakeEngine()
    worker = AdvisoryWorker(engine, advisory_repo, category="bottle")
    worker.start()
    try:
        assert worker.submit(_result(Verdict.FAIL, "insp-fail")) is True
        assert worker.drain(timeout=10.0) is True
    finally:
        worker.stop()

    report = advisory_repo.get_report("insp-fail")
    assert report is not None
    assert report.severity == Severity.MAJOR  # deterministic, echoed
    assert report.defect_classification == "scratch"
    # The metrology the station measured must reach the advisory layer.
    assert engine.seen[0].measurements[0].area_um2 == 1312.5


def test_worker_skips_passing_parts(advisory_repo: SqliteAdvisoryRepository) -> None:
    engine = _FakeEngine()
    worker = AdvisoryWorker(engine, advisory_repo)
    assert worker.submit(_result(Verdict.PASS, "insp-pass")) is False
    assert engine.seen == []


def test_worker_explains_review_parts(advisory_repo: SqliteAdvisoryRepository) -> None:
    worker = AdvisoryWorker(_FakeEngine(), advisory_repo)
    assert worker.submit(_result(Verdict.REVIEW, "insp-review")) is True


def test_worker_drops_rather_than_blocking_when_saturated(
    advisory_repo: SqliteAdvisoryRepository,
) -> None:
    """submit() is called from the inspection loop: it must never block."""
    worker = AdvisoryWorker(_FakeEngine(), advisory_repo, queue_size=1)
    assert worker.submit(_result(Verdict.FAIL, "insp-1")) is True
    assert worker.submit(_result(Verdict.FAIL, "insp-2")) is False


def test_worker_survives_a_failing_engine(advisory_repo: SqliteAdvisoryRepository) -> None:
    class _Exploding:
        def generate_report(self, evidence: InspectionEvidence) -> AdvisoryReport:
            raise RuntimeError("model crashed")

    worker = AdvisoryWorker(_Exploding(), advisory_repo)  # type: ignore[arg-type]
    worker.start()
    try:
        worker.submit(_result(Verdict.FAIL, "insp-boom"))
        assert worker.drain(timeout=10.0) is True
    finally:
        worker.stop()
    assert advisory_repo.get_report("insp-boom") is None


def test_worker_start_and_stop_are_idempotent(
    advisory_repo: SqliteAdvisoryRepository,
) -> None:
    worker = AdvisoryWorker(_FakeEngine(), advisory_repo)
    worker.stop()  # before start: no-op
    worker.start()
    worker.start()  # second start must not spawn a second thread
    worker.stop()
    worker.stop()


# -----------------------------------------------------------------------------
# Composition-root builders
# -----------------------------------------------------------------------------


def test_build_image_store_absent_by_default() -> None:
    assert build_image_store(_config()) is None


def test_build_image_store_from_config(tmp_path: Path) -> None:
    store = build_image_store(_config(IMAGE_STORE_DIR=str(tmp_path), IMAGE_STORE_MAX="5"))
    assert store is not None
    assert store.directory == tmp_path


def test_build_reject_dispatcher_absent_without_plc_host() -> None:
    assert build_reject_dispatcher(_config()) is None


def test_build_reject_dispatcher_degrades_when_plc_is_unreachable() -> None:
    """An unreachable PLC at boot disables dispatch; it never fails the boot."""
    assert build_reject_dispatcher(_config(PLC_HOST="127.0.0.1", PLC_PORT="1")) is None


def test_build_reject_dispatcher_connects_to_the_simulator(plc_server: Any) -> None:
    port = plc_server.server_address[1]
    dispatcher = build_reject_dispatcher(_config(PLC_HOST="127.0.0.1", PLC_PORT=str(port)))
    assert dispatcher is not None
    dispatcher.on_result(_result(Verdict.FAIL))


def test_build_advisory_worker_absent_unless_enabled(tmp_path: Path) -> None:
    _, session_factory = open_database(tmp_path / "a.db")
    assert build_advisory_worker(_config(), session_factory) is None
    assert build_advisory_worker(_config(ADVISORY_ENABLED="false"), session_factory) is None


def test_build_advisory_worker_when_enabled(tmp_path: Path) -> None:
    _, session_factory = open_database(tmp_path / "a.db")
    worker = build_advisory_worker(
        _config(ADVISORY_ENABLED="true", OLLAMA_TIMEOUT_S="1.0"), session_factory
    )
    assert isinstance(worker, AdvisoryWorker)


def test_chain_handlers_runs_every_consumer_in_order() -> None:
    calls: list[str] = []
    chained = chain_handlers(
        lambda r: calls.append("persist"),
        None,
        lambda r: calls.append("reject"),
        lambda r: calls.append("advisory"),
    )
    chained(_result())
    assert calls == ["persist", "reject", "advisory"]


def test_chain_handlers_with_nothing_configured_is_a_no_op() -> None:
    chain_handlers(None)(_result())


def test_default_lot_id_is_applied() -> None:
    assert _result().lot_id == DEFAULT_LOT_ID


# -----------------------------------------------------------------------------
# Manual inspection service (teach mode)
# -----------------------------------------------------------------------------


def test_manual_service_runs_the_pipeline_and_dispatches(tmp_path: Path) -> None:
    """A manually submitted sample must take the same path as a triggered
    part: same pipeline, same result consumers."""
    seen: list[InspectionResult] = []
    store = LocalImageStore(tmp_path)

    def factory(camera: Any) -> InspectionPipeline:
        return InspectionPipeline(
            camera, station_id="s1", recipe_ver="r1", image_store=store, lot_id="LOT-M"
        )

    service = ManualInspectionService(factory, seen.append)
    result = service.inspect(np.zeros((16, 16, 3), dtype=np.uint8), "manual-1")

    assert result.part_id == "manual-1"
    assert result.lot_id == "LOT-M"
    assert seen == [result]
    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_manual_service_without_a_handler_still_inspects() -> None:
    """The consumer chain is optional -- a bare service still produces a result."""

    def factory(camera: Any) -> InspectionPipeline:
        return InspectionPipeline(camera, station_id="s1", recipe_ver="r1")

    result = ManualInspectionService(factory).inspect(
        np.zeros((8, 8, 3), dtype=np.uint8), "manual-2"
    )
    assert result.part_id == "manual-2"


def test_manual_service_gives_each_call_its_own_camera() -> None:
    """Concurrent requests must not share per-cycle state."""
    cameras: list[Any] = []

    def factory(camera: Any) -> InspectionPipeline:
        cameras.append(camera)
        return InspectionPipeline(camera, station_id="s1", recipe_ver="r1")

    service = ManualInspectionService(factory)
    service.inspect(np.zeros((8, 8, 3), dtype=np.uint8), "a")
    service.inspect(np.zeros((8, 8, 3), dtype=np.uint8), "b")

    assert cameras[0] is not cameras[1]
    # Each driver is closed once its inspection finishes.
    assert all(not c.is_healthy() for c in cameras)


def test_build_manual_inspector_wires_persistence(tmp_path: Path) -> None:
    config = _config(DB_PATH=str(tmp_path / "station.db"), LOT_ID="LOT-BUILD")
    service, repository, session_factory = build_manual_inspector(config)

    result = service.inspect(np.zeros((16, 16, 3), dtype=np.uint8), "manual-built")

    assert result.lot_id == "LOT-BUILD"
    # The returned repository must serve the record the service just wrote.
    assert repository.get_result(result.inspection_id) is not None
    assert session_factory is not None
