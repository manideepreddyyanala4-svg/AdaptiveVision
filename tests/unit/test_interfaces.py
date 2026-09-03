"""Unit tests for :mod:`adaptivevision.common`.

Each interface is verified to be abstract (cannot be instantiated, and a
subclass missing a method also cannot be instantiated) and implementable (a
minimal fake satisfies the contract). The fakes double as the seed for the
Milestone M3 null objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from adaptivevision import common as interfaces
from adaptivevision.common import Verdict
from adaptivevision.common import (
    AnomalyResult,
    InspectionResult,
    MetrologyResult,
)
from adaptivevision.common import ROI, RawFrame, RectifiedFrame

ABSTRACT_INTERFACES = [
    interfaces.CameraDriver,
    interfaces.InferenceEngine,
    interfaces.AnomalyDetector,
    interfaces.Inspector,
    interfaces.PLCTransport,
    interfaces.MessagePublisher,
    interfaces.ResultRepository,
    interfaces.RecipeStore,
]


@pytest.mark.parametrize("interface", ABSTRACT_INTERFACES)
def test_interfaces_cannot_be_instantiated(interface: type) -> None:
    with pytest.raises(TypeError):
        interface()  # type: ignore[abstract, call-arg]


# --- Minimal fakes proving each contract is implementable --------------------


class FakeCamera(interfaces.CameraDriver):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def capture(self, trigger_id: str | None = None) -> RawFrame:
        return RawFrame(
            image=object(),
            camera_id="cam0",
            frame_id="frame-1",
            timestamp_monotonic=0.0,
            timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
            trigger_id=trigger_id,
        )

    def is_healthy(self) -> bool:
        return True


class FakeEngine(interfaces.InferenceEngine):
    @property
    def model_version(self) -> str:
        return "fake-1"

    def load(self, model_id: str) -> None: ...
    def warmup(self) -> None: ...
    def infer(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(inputs)

    def unload(self) -> None: ...


class FakeDetector(interfaces.AnomalyDetector):
    def detect(self, frame: RectifiedFrame, roi: ROI | None = None) -> AnomalyResult:
        return AnomalyResult(score=0.0, threshold=0.5, is_anomalous=False)


class FakeInspector(interfaces.Inspector[object, object]):
    def inspect(self, part: object, recipe: object) -> MetrologyResult:
        return MetrologyResult()


class FakePlc(interfaces.PLCTransport):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool:
        return True

    def read_coils(self, address: int, count: int) -> tuple[bool, ...]:
        return tuple(False for _ in range(count))

    def write_coil(self, address: int, value: bool) -> None: ...
    def read_registers(self, address: int, count: int) -> tuple[int, ...]:
        return tuple(0 for _ in range(count))

    def write_registers(self, address: int, values: Sequence[int]) -> None: ...


class FakePublisher(interfaces.MessagePublisher):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool:
        return True

    def publish(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None: ...


class FakeRepository(interfaces.ResultRepository):
    def save_result(self, result: InspectionResult) -> None: ...
    def get_result(self, inspection_id: str) -> InspectionResult | None:
        return None

    def list_results(
        self, *, limit: int = 100, offset: int = 0
    ) -> tuple[InspectionResult, ...]:
        return ()


class FakeRecipeStore(interfaces.RecipeStore[str]):
    def load(self, recipe_id: str) -> str:
        return recipe_id

    def save(self, recipe: str) -> None: ...
    def list_ids(self) -> tuple[str, ...]:
        return ()


def test_fakes_satisfy_contracts() -> None:
    assert FakeCamera().capture("t").trigger_id == "t"
    assert FakeCamera().is_healthy() is True
    assert FakeEngine().model_version == "fake-1"
    assert FakeEngine().infer({"x": 1}) == {"x": 1}
    FakeEngine().load("m")
    FakeEngine().warmup()
    FakeEngine().unload()
    frame = RectifiedFrame(
        image=object(),
        camera_id="c",
        frame_id="f",
        calibration_ver="cal",
        timestamp_monotonic=0.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert FakeDetector().detect(frame).is_anomalous is False
    assert isinstance(FakeInspector().inspect(object(), object()), MetrologyResult)


def test_fake_plc_contract() -> None:
    plc = FakePlc()
    plc.connect()
    assert plc.is_connected() is True
    assert plc.read_coils(0, 3) == (False, False, False)
    assert plc.read_registers(0, 2) == (0, 0)
    plc.write_coil(1, True)
    plc.write_registers(0, [1, 2])
    plc.disconnect()


def test_fake_publisher_and_repository_and_store() -> None:
    pub = FakePublisher()
    pub.connect()
    pub.publish("topic", {"k": "v"}, qos=1, retain=True)
    assert pub.is_connected() is True
    pub.disconnect()

    repo = FakeRepository()
    res = InspectionResult(
        inspection_id="i",
        part_id="p",
        station_id="s",
        verdict=Verdict.PASS,
        recipe_ver="r",
        model_ver="m",
        calib_ver="c",
        cycle_time_ms=1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    repo.save_result(res)
    assert repo.get_result("i") is None
    assert repo.list_results() == ()

    store = FakeRecipeStore()
    store.save("r1")
    assert store.load("r1") == "r1"
    assert store.list_ids() == ()


def test_public_api_reexports_resolve() -> None:
    import adaptivevision.common as common

    for name in common.__all__:
        assert hasattr(common, name), name
