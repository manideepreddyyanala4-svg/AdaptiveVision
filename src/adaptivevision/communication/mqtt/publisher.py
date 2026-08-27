"""MQTT message publisher (Milestone M12).

The publisher implements the :class:`~adaptivevision.common.interfaces.MessagePublisher`
seam. The actual broker protocol is delegated to a pluggable
:class:`MqttClient` so the publisher stays free of external dependencies and can
be exercised with a fake client in tests. Payloads are serialized to JSON before
publishing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from adaptivevision.common.errors import CommsError
from adaptivevision.common.interfaces import MessagePublisher


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
    """A :class:`MessagePublisher` backed by an MQTT client.

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
        """Raise :class:`CommsError` when the publisher is not connected."""
        if not self._connected:
            raise CommsError("MQTT publisher is not connected")
