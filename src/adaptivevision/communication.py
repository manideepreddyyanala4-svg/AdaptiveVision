"""Factory-protocol messaging: PLC (Modbus TCP) and MQTT.

Both transports implement their respective seam
(:class:`~adaptivevision.common.PLCTransport`,
:class:`~adaptivevision.common.MessagePublisher`) by delegating the actual
wire protocol to a pluggable low-level client (:class:`ModbusClient`,
:class:`MqttClient`), so this module stays free of external networking
dependencies and can be exercised with a fake client in tests. A real client
can be supplied at the composition root without changing either transport.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import struct
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from adaptivevision.common import (
    CommsError,
    InspectionResult,
    MessagePublisher,
    PLCTransport,
    Verdict,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Modbus TCP transport for PLC register / coil access
# =============================================================================


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
    """A :class:`~adaptivevision.common.PLCTransport` backed by a Modbus TCP client.

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
        """Raise :class:`~adaptivevision.common.CommsError` when the transport is not connected."""
        if not self._connected:
            raise CommsError("PLC transport is not connected")


# =============================================================================
# MQTT message publisher
# =============================================================================


class MqttClient(Protocol):
    """Low-level MQTT client used by the publisher.

    Implementations are responsible for the MQTT connect / publish framing.
    """

    def connect(self) -> None:
        """Open the broker connection."""

    def disconnect(self) -> None:
        """Close the broker connection."""

    def publish(self, topic: str, payload: bytes, qos: int, retain: bool) -> None:
        """Publish a raw payload to a topic."""


class MqttPublisher(MessagePublisher):
    """A :class:`~adaptivevision.common.MessagePublisher` backed by an MQTT client.

    Args:
        client: The low-level MQTT client to delegate to.
    """

    def __init__(self, client: MqttClient) -> None:
        """Initialize the publisher."""
        self._client = client
        self._connected = False

    def connect(self) -> None:
        """Establish the broker connection.

        Raises:
            CommsError: If the connection cannot be established.
        """
        try:
            self._client.connect()
        except Exception as exc:
            raise CommsError(f"MQTT connect failed: {exc}") from exc
        self._connected = True

    def disconnect(self) -> None:
        """Close the broker connection."""
        try:
            self._client.disconnect()
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        """Return ``True`` if the publisher is connected."""
        return self._connected

    def publish(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        """Publish a payload to a topic.

        Args:
            topic: Destination topic.
            payload: JSON-serializable message body.
            qos: Delivery quality-of-service level.
            retain: Whether the broker should retain the message.

        Raises:
            CommsError: On publish failure.
        """
        self._require_connected()
        try:
            body = json.dumps(payload).encode("utf-8")
            self._client.publish(topic, body, qos, retain)
        except Exception as exc:
            raise CommsError(f"MQTT publish failed: {exc}") from exc

    def _require_connected(self) -> None:
        """Raise :class:`~adaptivevision.common.CommsError` when the publisher is not connected."""
        if not self._connected:
            raise CommsError("MQTT publisher is not connected")


# =============================================================================
# Verdict -> PLC reject dispatch
#
# Turns a finished inspection into the one physical action the line cares
# about: energizing the reject coil that fires the pneumatic pusher. Runs as
# an ``on_result`` consumer, off the inspection critical path, and never
# propagates a communication failure -- a PLC that is down must not stop the
# station from inspecting and recording parts.
# =============================================================================

#: Default discrete coil the reject actuator is wired to.
DEFAULT_REJECT_COIL = 3


class RejectDispatcher:
    """Asserts a PLC reject coil for every failing part.

    Args:
        transport: The connected PLC transport to write through.
        coil_address: Discrete coil driving the reject actuator.
        reject_verdicts: Verdicts that should energize the coil. Defaults to
            ``FAIL`` only -- ``REVIEW`` parts are routed to a human, not
            physically ejected.
    """

    def __init__(
        self,
        transport: PLCTransport,
        *,
        coil_address: int = DEFAULT_REJECT_COIL,
        reject_verdicts: tuple[Verdict, ...] = (Verdict.FAIL,),
    ) -> None:
        """Initialize the dispatcher."""
        self._transport = transport
        self._coil_address = coil_address
        self._reject_verdicts = reject_verdicts

    def on_result(self, result: InspectionResult) -> None:
        """Assert or clear the reject coil for ``result``.

        Communication failures are logged and swallowed: the verdict is
        already recorded, and a PLC outage is an operations problem, not a
        reason to fault the inspection loop.

        Args:
            result: The completed inspection result.
        """
        reject = result.verdict in self._reject_verdicts
        try:
            self._transport.write_coil(self._coil_address, reject)
        except CommsError as exc:
            logger.error(
                "PLC reject dispatch failed",
                extra={
                    "inspection_id": result.inspection_id,
                    "part_id": result.part_id,
                    "verdict": result.verdict.value,
                    "coil": self._coil_address,
                    "error": str(exc),
                },
            )
            return
        if reject:
            logger.warning(
                "Reject asserted",
                extra={
                    "inspection_id": result.inspection_id,
                    "part_id": result.part_id,
                    "coil": self._coil_address,
                },
            )


# =============================================================================
# Dependency-free Modbus TCP client
#
# Implements the ModbusClient protocol directly over a socket, so a station
# can drive a real PLC (or scripts/mock_plc.py) without adding a networking
# dependency. Covers only the four function codes the protocol declares.
# =============================================================================

#: MBAP header: transaction id, protocol id, length, unit id.
_MBAP = struct.Struct(">HHHB")

_FN_READ_COILS = 0x01
_FN_READ_HOLDING = 0x03
_FN_WRITE_COIL = 0x05
_FN_WRITE_REGISTERS = 0x10

#: Value meaning "coil on" in a write-single-coil request.
_COIL_ON = 0xFF00


class SocketModbusClient:
    """A :class:`ModbusClient` speaking Modbus TCP over a plain socket.

    Args:
        host: PLC address.
        port: PLC port (502 in the field, 5020 for the bench simulator).
        unit_id: Modbus unit identifier.
        timeout: Socket timeout in seconds. Kept short: the station must
            never block on an unresponsive PLC.
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        *,
        unit_id: int = 1,
        timeout: float = 1.0,
    ) -> None:
        """Initialize the client without connecting."""
        self._host = host
        self._port = port
        self._unit_id = unit_id
        self._timeout = timeout
        self._socket: socket.socket | None = None
        self._transaction = 0

    def connect(self) -> None:
        """Open the TCP connection."""
        self._socket = socket.create_connection((self._host, self._port), timeout=self._timeout)

    def close(self) -> None:
        """Close the TCP connection."""
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.close()
            self._socket = None

    def read_coils(self, address: int, count: int) -> list[bool]:
        """Read ``count`` coils starting at ``address``."""
        payload = self._request(struct.pack(">BHH", _FN_READ_COILS, address, count))
        packed = payload[2:]
        return [bool(packed[i // 8] >> (i % 8) & 1) for i in range(count)]

    def write_coil(self, address: int, value: bool) -> None:
        """Write a single coil at ``address``."""
        self._request(
            struct.pack(">BHH", _FN_WRITE_COIL, address, _COIL_ON if value else 0x0000)
        )

    def read_holding_registers(self, address: int, count: int) -> list[int]:
        """Read ``count`` holding registers starting at ``address``."""
        payload = self._request(struct.pack(">BHH", _FN_READ_HOLDING, address, count))
        return list(struct.unpack(">" + "H" * count, payload[2 : 2 + count * 2]))

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        """Write holding registers starting at ``address``."""
        body = b"".join(struct.pack(">H", v) for v in values)
        self._request(
            struct.pack(">BHHB", _FN_WRITE_REGISTERS, address, len(values), len(body)) + body
        )

    def _request(self, pdu: bytes) -> bytes:
        """Send one request PDU and return the response PDU.

        Raises:
            OSError: On any socket failure. ``ModbusTcpTransport`` wraps this
                into a :class:`~adaptivevision.common.CommsError`.
        """
        if self._socket is None:
            msg = "SocketModbusClient is not connected"
            raise OSError(msg)
        self._transaction = (self._transaction + 1) % 0x10000
        self._socket.sendall(_MBAP.pack(self._transaction, 0, len(pdu) + 1, self._unit_id) + pdu)
        header = self._read_exactly(_MBAP.size)
        _, _, length, _ = _MBAP.unpack(header)
        response = self._read_exactly(length - 1)
        if response[0] & 0x80:
            msg = f"Modbus exception 0x{response[1]:02X} for function 0x{pdu[0]:02X}"
            raise OSError(msg)
        return response

    def _read_exactly(self, count: int) -> bytes:
        """Read exactly ``count`` bytes from the socket."""
        assert self._socket is not None
        chunks = bytearray()
        while len(chunks) < count:
            chunk = self._socket.recv(count - len(chunks))
            if not chunk:
                msg = "PLC closed the connection"
                raise OSError(msg)
            chunks.extend(chunk)
        return bytes(chunks)
