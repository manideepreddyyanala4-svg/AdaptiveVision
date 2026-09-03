"""Unit tests for communication.py: Modbus PLC transport and MQTT publisher."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from adaptivevision.common import CommsError
from adaptivevision.communication import ModbusTcpTransport, MqttPublisher

# -----------------------------------------------------------------------------
# Modbus TCP PLC transport
# -----------------------------------------------------------------------------

class _FakePlcClient:
    def __init__(self) -> None:
        self.connected = False
        self.coils: dict[int, bool] = {}
        self.registers: dict[int, int] = {}
        self.fail_connect = False
        self.fail_read = False

    def connect(self) -> None:
        if self.fail_connect:
            raise OSError("connection refused")
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def read_coils(self, address: int, count: int) -> list[bool]:
        if self.fail_read:
            raise OSError("read timeout")
        return [self.coils.get(address + i, False) for i in range(count)]

    def write_coil(self, address: int, value: bool) -> None:
        self.coils[address] = value

    def read_holding_registers(self, address: int, count: int) -> list[int]:
        if self.fail_read:
            raise OSError("read timeout")
        return [self.registers.get(address + i, 0) for i in range(count)]

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        for i, value in enumerate(values):
            self.registers[address + i] = value


def test_plc_connect_and_disconnect() -> None:
    client = _FakePlcClient()
    transport = ModbusTcpTransport(client)
    assert transport.is_connected() is False
    transport.connect()
    assert transport.is_connected() is True
    transport.disconnect()
    assert transport.is_connected() is False


def test_plc_connect_failure_raises_comms_error() -> None:
    client = _FakePlcClient()
    client.fail_connect = True
    transport = ModbusTcpTransport(client)
    with pytest.raises(CommsError):
        transport.connect()


def test_read_and_write_coils() -> None:
    client = _FakePlcClient()
    transport = ModbusTcpTransport(client)
    transport.connect()
    transport.write_coil(10, True)
    assert transport.read_coils(10, 1) == (True,)
    assert transport.read_coils(10, 3) == (True, False, False)


def test_read_and_write_registers() -> None:
    client = _FakePlcClient()
    transport = ModbusTcpTransport(client)
    transport.connect()
    transport.write_registers(20, [1, 2, 3])
    assert transport.read_registers(20, 3) == (1, 2, 3)


def test_operations_require_connection() -> None:
    client = _FakePlcClient()
    transport = ModbusTcpTransport(client)
    with pytest.raises(CommsError):
        transport.read_coils(0, 1)
    with pytest.raises(CommsError):
        transport.write_coil(0, True)
    with pytest.raises(CommsError):
        transport.read_registers(0, 1)
    with pytest.raises(CommsError):
        transport.write_registers(0, [1])


def test_client_failure_translated_to_comms_error() -> None:
    client = _FakePlcClient()
    client.fail_read = True
    transport = ModbusTcpTransport(client)
    transport.connect()
    with pytest.raises(CommsError):
        transport.read_coils(0, 1)
    with pytest.raises(CommsError):
        transport.read_registers(0, 1)


# -----------------------------------------------------------------------------
# MQTT publisher
# -----------------------------------------------------------------------------

class _FakeMqttClient:
    def __init__(self) -> None:
        self.connected = False
        self.messages: list[tuple[str, bytes, int, bool]] = []
        self.fail_connect = False
        self.fail_publish = False

    def connect(self) -> None:
        if self.fail_connect:
            raise OSError("broker unreachable")
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def publish(self, topic: str, payload: bytes, qos: int, retain: bool) -> None:
        if self.fail_publish:
            raise OSError("publish timeout")
        self.messages.append((topic, payload, qos, retain))


def test_mqtt_connect_and_disconnect() -> None:
    client = _FakeMqttClient()
    publisher = MqttPublisher(client)
    assert publisher.is_connected() is False
    publisher.connect()
    assert publisher.is_connected() is True
    publisher.disconnect()
    assert publisher.is_connected() is False


def test_mqtt_connect_failure_raises_comms_error() -> None:
    client = _FakeMqttClient()
    client.fail_connect = True
    publisher = MqttPublisher(client)
    with pytest.raises(CommsError):
        publisher.connect()


def test_publish_serializes_payload_to_json() -> None:
    client = _FakeMqttClient()
    publisher = MqttPublisher(client)
    publisher.connect()
    publisher.publish("parts/1", {"verdict": "PASS", "score": 0.9})
    assert len(client.messages) == 1
    topic, payload, qos, retain = client.messages[0]
    assert topic == "parts/1"
    assert json.loads(payload) == {"verdict": "PASS", "score": 0.9}
    assert qos == 0
    assert retain is False


def test_publish_forwards_qos_and_retain() -> None:
    client = _FakeMqttClient()
    publisher = MqttPublisher(client)
    publisher.connect()
    publisher.publish("parts/1", {"ok": True}, qos=1, retain=True)
    _, _, qos, retain = client.messages[0]
    assert qos == 1
    assert retain is True


def test_publish_requires_connection() -> None:
    client = _FakeMqttClient()
    publisher = MqttPublisher(client)
    with pytest.raises(CommsError):
        publisher.publish("parts/1", {"ok": True})


def test_publish_failure_translated_to_comms_error() -> None:
    client = _FakeMqttClient()
    client.fail_publish = True
    publisher = MqttPublisher(client)
    publisher.connect()
    with pytest.raises(CommsError):
        publisher.publish("parts/1", {"ok": True})
