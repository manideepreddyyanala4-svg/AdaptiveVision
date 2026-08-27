"""Structured JSON logging setup for AdaptiveVision.

Logging is a cross-cutting concern: every subsystem emits structured records
that are consumed by the dashboard and the operator console. This module owns
the single, shared logging configuration so all components produce consistent
JSON lines.

It exposes:

* :func:`configure_logging` - install a JSON formatter on the root logger,
* :func:`get_logger` - return a named logger,
* :func:`set_correlation_id` / :func:`get_correlation_id` /
  :func:`clear_correlation_id` - manage the per-thread correlation id,
* :func:`correlation_context` - a context manager that scopes a correlation id.

The correlation id is carried on every record so an operator can trace a single
inspection across the acquisition, orchestration, and persistence logs.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, TextIO

#: Sentinel correlation id used when no explicit id has been set.
DEFAULT_CORRELATION_ID = ""

#: Per-thread correlation id attached to every log record.
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=DEFAULT_CORRELATION_ID
)

#: Standard :class:`logging.LogRecord` attributes that are not "extra" fields.
_STANDARD_ATTRIBUTES = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a JSON line."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        payload.update(_extra_fields(record))
        return json.dumps(payload, sort_keys=True)


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return the non-standard attributes attached to a log record.

    ``extra`` keyword arguments passed to a logging call are merged into the
    record's ``__dict__``; this helper extracts everything that is not a
    standard :class:`logging.LogRecord` attribute.

    Args:
        record: The log record to inspect.

    Returns:
        A mapping of extra field name to value.
    """
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_ATTRIBUTES
    }


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (typically the module path).

    Returns:
        A :class:`logging.Logger` configured to emit JSON lines.
    """
    return logging.getLogger(name)


def configure_logging(
    *,
    level: str | int = "INFO",
    stream: TextIO | None = None,
    force: bool = True,
) -> None:
    """Configure the root logger to emit JSON lines.

    Args:
        level: Root logging level name (for example ``"INFO"``) or a numeric
            level constant.

        stream: Output stream. Defaults to standard output.
        force: When ``True``, replace existing handlers; when ``False``, append
            a new handler.


    Raises:
        ValueError: If ``level`` is not a recognized logging level.
    """
    resolved = _resolve_level(level)
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)

    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    if force:
        root.handlers[:] = [handler]
    else:
        root.addHandler(handler)
    root.setLevel(resolved)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation id for the current thread.

    Args:
        correlation_id: Identifier to attach to subsequent log records.
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str:
    """Return the correlation id for the current thread."""
    return _correlation_id.get()


def clear_correlation_id() -> None:
    """Reset the correlation id for the current thread to the default."""
    _correlation_id.set(DEFAULT_CORRELATION_ID)


@contextlib.contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Scope a correlation id for the duration of the ``with`` block.

    The previous correlation id is restored on exit, including on exception.

    Args:
        correlation_id: Identifier to attach to records within the block.

    Yields:
        None.
    """
    previous = _correlation_id.get()
    _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.set(previous)


def _resolve_level(level: str | int) -> int:
    """Resolve a level name or numeric value to a logging level constant.

    Args:
        level: Level name (for example ``"INFO"``) or numeric constant.

    Returns:
        The resolved numeric logging level.

    Raises:
        ValueError: If ``level`` is not a recognized logging level.
    """
    if isinstance(level, int):
        return level
    normalized = level.upper()
    resolved = logging.getLevelNamesMapping().get(normalized)
    if resolved is None:
        msg = f"Unknown log level: {level!r}"
        raise ValueError(msg)
    return resolved
