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

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from adaptivevision.common import CommsError, MessagePublisher, PLCTransport

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
    """A :class:`~adaptivevision.common.PLCTransport` backed by a Modbus TCP
    client.

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
        """Raise :class:`~adaptivevision.common.CommsError` when the transport
        is not connected."""
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
    """A :class:`~adaptivevision.common.MessagePublisher` backed by an MQTT
    client.

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
        """Raise :class:`~adaptivevision.common.CommsError` when the publisher
        is not connected."""
        if not self._connected:
            raise CommsError("MQTT publisher is not connected")
