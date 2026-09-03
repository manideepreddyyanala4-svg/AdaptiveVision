"""Modbus TCP transport for PLC register / coil access (Milestone M11).

The transport implements the :class:`~adaptivevision.common.interfaces.PLCTransport`
seam. The actual wire protocol is delegated to a pluggable
:class:`ModbusClient` so the transport stays free of external dependencies and
can be exercised with a fake client in tests. A real client can be supplied at
the composition root without changing the transport.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from adaptivevision.common.errors import CommsError
from adaptivevision.common.interfaces import PLCTransport


class ModbusClient(Protocol):
    """Low-level Modbus TCP client used by the transport.

    Implementations are responsible for the MBAP framing and function codes.
    """

    def connect(self) -> None:
        """Open the TCP connection to the PLC."""

    def close(self) -> None:
        """Close the TCP connection."""

    def read_coils(self, address: int, count: int) -> list[bool]:
        """Read ``count`` coils starting at ``address``."""

    def write_coil(self, address: int, value: bool) -> None:
        """Write a single coil at ``address``."""

    def read_holding_registers(self, address: int, count: int) -> list[int]:
        """Read ``count`` holding registers starting at ``address``."""

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        """Write holding registers starting at ``address``."""


class ModbusTcpTransport(PLCTransport):
    """A :class:`PLCTransport` backed by a Modbus TCP client.

    Args:
        client: The low-level Modbus client to delegate to.
    """

    def __init__(self, client: ModbusClient) -> None:
        """Initialize the transport."""
        self._client = client
        self._connected = False

    def connect(self) -> None:
        """Establish the transport connection.

        Raises:
            CommsError: If the connection cannot be established.
        """
        try:
            self._client.connect()
        except Exception as exc:
            raise CommsError(f"PLC connect failed: {exc}") from exc
        self._connected = True

    def disconnect(self) -> None:
        """Close the transport connection."""
        try:
            self._client.close()
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        """Return ``True`` if the transport is connected."""
        return self._connected

    def read_coils(self, address: int, count: int) -> tuple[bool, ...]:
        """Read ``count`` coils starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """
        self._require_connected()
        try:
            return tuple(self._client.read_coils(address, count))
        except Exception as exc:
            raise CommsError(f"PLC read_coils failed: {exc}") from exc

    def write_coil(self, address: int, value: bool) -> None:
        """Write a single coil.

        Raises:
            CommsError: On communication failure.
        """
        self._require_connected()
        try:
            self._client.write_coil(address, value)
        except Exception as exc:
            raise CommsError(f"PLC write_coil failed: {exc}") from exc

    def read_registers(self, address: int, count: int) -> tuple[int, ...]:
        """Read ``count`` holding registers starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """
        self._require_connected()
        try:
            return tuple(self._client.read_holding_registers(address, count))
        except Exception as exc:
            raise CommsError(f"PLC read_registers failed: {exc}") from exc

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        """Write holding registers starting at ``address``.

        Raises:
            CommsError: On communication failure.
        """
        self._require_connected()
        try:
            self._client.write_registers(address, values)
        except Exception as exc:
            raise CommsError(f"PLC write_registers failed: {exc}") from exc

    def _require_connected(self) -> None:
        """Raise :class:`CommsError` when the transport is not connected."""
        if not self._connected:
            raise CommsError("PLC transport is not connected")
