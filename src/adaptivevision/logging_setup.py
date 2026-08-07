"""Structured JSON logging bootstrap for AdaptiveVision.

This module is the single entry point for configuring application-wide logging.
It emits exactly one JSON object per log record on a single line, which makes
logs machine-parseable by downstream aggregators (for example Loki or ELK), as
described in *Architecture Specification v1.0*, Section 12 (Logging
Architecture).

Responsibilities at Milestone M0:

* Provide a JSON log formatter with a stable field schema.
* Inject a context-scoped ``correlation_id`` into every record via a
  :class:`contextvars.ContextVar`, so a single unit of work (later, an
  ``inspection_id``) can be traced across all of its log lines.
* Offer an idempotent :func:`configure_logging` bootstrap.

This module deliberately owns only the logging *mechanism*. The domain *values*
placed into the correlation context (such as inspection identifiers) are the
responsibility of later milestones and are not implemented here.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any, TextIO

#: Sentinel used when no correlation id has been bound to the current context.
DEFAULT_CORRELATION_ID = "-"

_CORRELATION_ID: ContextVar[str] = ContextVar(
    "adaptivevision_correlation_id",
    default=DEFAULT_CORRELATION_ID,
)

# Attributes already present on a standard LogRecord (plus a few added during
# formatting). Any *other* attribute on a record is treated as a caller-supplied
# ``extra`` field and merged into the JSON payload.
_RESERVED_ATTRS: frozenset[str] = frozenset(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
    "correlation_id",
}


def get_correlation_id() -> str:
    """Return the correlation id bound to the current context."""
    return _CORRELATION_ID.get()


def set_correlation_id(correlation_id: str) -> Token[str]:
    """Bind ``correlation_id`` to the current context.

    Args:
        correlation_id: Identifier to attach to subsequent log records.

    Returns:
        A token that may be passed to :meth:`contextvars.ContextVar.reset`
        to restore the previous value.
    """
    return _CORRELATION_ID.set(correlation_id)


def clear_correlation_id() -> None:
    """Reset the correlation id to :data:`DEFAULT_CORRELATION_ID`."""
    _CORRELATION_ID.set(DEFAULT_CORRELATION_ID)


@contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Bind ``correlation_id`` for the duration of the ``with`` block.

    The previous correlation id is restored on exit, even if an exception is
    raised inside the block.

    Args:
        correlation_id: Identifier to attach to log records emitted inside the
            block.

    Yields:
        ``None``.
    """
    token = _CORRELATION_ID.set(correlation_id)
    try:
        yield
    finally:
        _CORRELATION_ID.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Attach the current correlation id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject ``correlation_id`` onto ``record`` and always allow it."""
        record.correlation_id = get_correlation_id()
        return True


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON objects with a stable schema.

    The emitted object always contains ``timestamp``, ``level``, ``logger``,
    ``message`` and ``correlation_id``. Source-location fields and any
    caller-supplied ``extra`` values are included as additional keys.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON string."""
        payload: dict[str, Any] = {
            "timestamp": self._format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", DEFAULT_CORRELATION_ID),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge caller-supplied ``extra`` fields without clobbering the schema.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        """Return the record's creation time as an ISO-8601 UTC string."""
        created = datetime.fromtimestamp(record.created, tz=UTC)
        return created.isoformat(timespec="milliseconds")


def configure_logging(
    *,
    level: int | str = "INFO",
    stream: TextIO | None = None,
    force: bool = True,
) -> None:
    """Configure application-wide structured logging.

    Installs a single JSON stream handler on the root logger. The call is
    idempotent: by default it removes existing root handlers first, so repeated
    invocations never accumulate duplicate handlers.

    Args:
        level: Minimum level to emit, as a name (``"INFO"``) or numeric value.
        stream: Target stream. Defaults to :data:`sys.stdout`.
        force: When ``True`` (default), remove existing root handlers before
            installing the JSON handler.
    """
    root = logging.getLogger()

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    target: TextIO = stream if stream is not None else sys.stdout
    json_handler: logging.Handler = logging.StreamHandler(target)
    json_handler.setFormatter(JsonLogFormatter())
    json_handler.addFilter(CorrelationIdFilter())

    root.addHandler(json_handler)
    root.setLevel(_coerce_level(level))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    A thin wrapper over :func:`logging.getLogger` so callers depend on this
    module rather than the standard library directly.

    Args:
        name: Dotted logger name, conventionally the module ``__name__``.

    Returns:
        The requested :class:`logging.Logger`.
    """
    return logging.getLogger(name)


def _coerce_level(level: int | str) -> int:
    """Convert a level name or numeric level to a numeric logging level.

    Args:
        level: Either an integer level or a level name such as ``"INFO"``.

    Returns:
        The numeric logging level.

    Raises:
        ValueError: If ``level`` is a string that is not a known level name.
    """
    if isinstance(level, int):
        return level

    mapping = logging.getLevelNamesMapping()
    resolved = mapping.get(level.upper())
    if resolved is None:
        msg = f"Unknown log level: {level!r}"
        raise ValueError(msg)
    return resolved
