"""Unit tests for M11 Modbus TCP PLC transport."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from adaptivevision.common import CommsError
from adaptivevision.communication import ModbusTcpTransport


class _FakeClient:
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


def test_connect_and_disconnect() -> None:
    client = _FakeClient()
    transport = ModbusTcpTransport(client)
    assert transport.is_connected() is False
    transport.connect()
    assert transport.is_connected() is True
    transport.disconnect()
    assert transport.is_connected() is False


def test_connect_failure_raises_comms_error() -> None:
    client = _FakeClient()
    client.fail_connect = True
    transport = ModbusTcpTransport(client)
    with pytest.raises(CommsError):
        transport.connect()


def test_read_and_write_coils() -> None:
    client = _FakeClient()
    transport = ModbusTcpTransport(client)
    transport.connect()
    transport.write_coil(10, True)
    assert transport.read_coils(10, 1) == (True,)
    assert transport.read_coils(10, 3) == (True, False, False)


def test_read_and_write_registers() -> None:
    client = _FakeClient()
    transport = ModbusTcpTransport(client)
    transport.connect()
    transport.write_registers(20, [1, 2, 3])
    assert transport.read_registers(20, 3) == (1, 2, 3)


def test_operations_require_connection() -> None:
    client = _FakeClient()
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
    client = _FakeClient()
    client.fail_read = True
    transport = ModbusTcpTransport(client)
    transport.connect()
    with pytest.raises(CommsError):
        transport.read_coils(0, 1)
    with pytest.raises(CommsError):
        transport.read_registers(0, 1)
