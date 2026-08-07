"""Unit tests for :mod:`adaptivevision.logging_setup`."""

from __future__ import annotations

import io
import json
import logging

import pytest

from adaptivevision import logging_setup

REQUIRED_FIELDS = ("timestamp", "level", "logger", "message", "correlation_id")


def _last_json_line(stream: io.StringIO) -> dict[str, object]:
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "expected at least one log line"
    return json.loads(lines[-1])


def test_emits_valid_json_with_required_fields(log_stream: io.StringIO) -> None:
    logging_setup.get_logger("test.logger").info("hello world")
    payload = _last_json_line(log_stream)

    for field in REQUIRED_FIELDS:
        assert field in payload
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["logger"] == "test.logger"


def test_timestamp_is_iso8601_utc(log_stream: io.StringIO) -> None:
    logging_setup.get_logger("t").info("m")
    payload = _last_json_line(log_stream)
    assert isinstance(payload["timestamp"], str)
    assert payload["timestamp"].endswith("+00:00")


def test_correlation_id_defaults_to_sentinel(log_stream: io.StringIO) -> None:
    logging_setup.get_logger("t").info("m")
    payload = _last_json_line(log_stream)
    assert payload["correlation_id"] == logging_setup.DEFAULT_CORRELATION_ID


def test_set_correlation_id_populates_records(log_stream: io.StringIO) -> None:
    logging_setup.set_correlation_id("insp-42")
    logging_setup.get_logger("t").info("m")
    payload = _last_json_line(log_stream)
    assert payload["correlation_id"] == "insp-42"


def test_correlation_context_sets_and_restores(log_stream: io.StringIO) -> None:
    logger = logging_setup.get_logger("t")
    with logging_setup.correlation_context("abc-123"):
        logger.info("inside")
    logger.info("outside")

    payloads = [json.loads(raw) for raw in log_stream.getvalue().splitlines() if raw.strip()]
    assert payloads[-2]["correlation_id"] == "abc-123"
    assert payloads[-1]["correlation_id"] == logging_setup.DEFAULT_CORRELATION_ID


def test_correlation_context_restores_on_exception(log_stream: io.StringIO) -> None:
    with pytest.raises(ValueError, match="boom"), logging_setup.correlation_context("x-1"):
        raise ValueError("boom")
    assert logging_setup.get_correlation_id() == logging_setup.DEFAULT_CORRELATION_ID


def test_extra_fields_are_included(log_stream: io.StringIO) -> None:
    logging_setup.get_logger("t").info("m", extra={"verdict": "PASS", "score": 0.12})
    payload = _last_json_line(log_stream)
    assert payload["verdict"] == "PASS"
    assert payload["score"] == pytest.approx(0.12)


def test_exception_is_serialized(log_stream: io.StringIO) -> None:
    logger = logging_setup.get_logger("t")
    try:
        raise ValueError("kaboom")
    except ValueError:
        logger.exception("operation failed")
    payload = _last_json_line(log_stream)
    assert "exception" in payload
    assert "ValueError" in str(payload["exception"])


def test_stack_info_is_serialized(log_stream: io.StringIO) -> None:
    logging_setup.get_logger("t").info("m", stack_info=True)
    payload = _last_json_line(log_stream)
    assert "stack" in payload


def test_configure_logging_is_idempotent() -> None:
    logging_setup.configure_logging()
    logging_setup.configure_logging()
    root = logging.getLogger()
    json_handlers = [
        h for h in root.handlers if isinstance(h.formatter, logging_setup.JsonLogFormatter)
    ]
    assert len(json_handlers) == 1


def test_configure_logging_force_false_appends_handler() -> None:
    stream = io.StringIO()
    logging_setup.configure_logging(level="INFO", stream=stream, force=True)
    logging_setup.configure_logging(level="INFO", stream=stream, force=False)
    root = logging.getLogger()
    json_handlers = [
        h for h in root.handlers if isinstance(h.formatter, logging_setup.JsonLogFormatter)
    ]
    assert len(json_handlers) == 2


def test_level_filtering_suppresses_below_threshold() -> None:
    stream = io.StringIO()
    logging_setup.configure_logging(level="WARNING", stream=stream)
    logger = logging_setup.get_logger("t")
    logger.info("suppressed message")
    logger.warning("shown message")
    output = stream.getvalue()
    assert "suppressed message" not in output
    assert "shown message" in output


def test_configure_logging_accepts_numeric_level() -> None:
    stream = io.StringIO()
    logging_setup.configure_logging(level=logging.ERROR, stream=stream)
    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        logging_setup.configure_logging(level="NOPE")


def test_get_logger_returns_named_logger() -> None:
    assert logging_setup.get_logger("a.b.c").name == "a.b.c"


def test_clear_correlation_id_resets_to_default() -> None:
    logging_setup.set_correlation_id("temp")
    logging_setup.clear_correlation_id()
    assert logging_setup.get_correlation_id() == logging_setup.DEFAULT_CORRELATION_ID
