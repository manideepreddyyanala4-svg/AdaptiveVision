"""Unit tests for M12 MQTT message publisher."""

from __future__ import annotations

import json

import pytest

from adaptivevision.common.errors import CommsError
from adaptivevision.communication.mqtt import MqttPublisher


class _FakeClient:
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


def test_connect_and_disconnect() -> None:
    client = _FakeClient()
    publisher = MqttPublisher(client)
    assert publisher.is_connected() is False
    publisher.connect()
    assert publisher.is_connected() is True
    publisher.disconnect()
    assert publisher.is_connected() is False


def test_connect_failure_raises_comms_error() -> None:
    client = _FakeClient()
    client.fail_connect = True
    publisher = MqttPublisher(client)
    with pytest.raises(CommsError):
        publisher.connect()


def test_publish_serializes_payload_to_json() -> None:
    client = _FakeClient()
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
    client = _FakeClient()
    publisher = MqttPublisher(client)
    publisher.connect()
    publisher.publish("parts/1", {"ok": True}, qos=1, retain=True)
    _, _, qos, retain = client.messages[0]
    assert qos == 1
    assert retain is True


def test_publish_requires_connection() -> None:
    client = _FakeClient()
    publisher = MqttPublisher(client)
    with pytest.raises(CommsError):
        publisher.publish("parts/1", {"ok": True})


def test_publish_failure_translated_to_comms_error() -> None:
    client = _FakeClient()
    client.fail_publish = True
    publisher = MqttPublisher(client)
    publisher.connect()
    with pytest.raises(CommsError):
        publisher.publish("parts/1", {"ok": True})
