"""Domain identifier generation.

Produces collision-resistant, time-ordered identifiers for inspections, parts,
frames, and traces. Each identifier is ``{prefix}-{ms:013d}-{rand}`` where
``ms`` is the zero-padded Unix time in milliseconds, so lexical sort
approximates creation order (to millisecond precision) and the random suffix
guarantees uniqueness within a millisecond.

This module is deliberately independent of logging and configuration (frozen
decision 8): it *generates* identifiers; binding one into the logging
correlation context is the orchestration layer's responsibility (Milestone M3).
"""

from __future__ import annotations

import time
import uuid

_MS_WIDTH = 13
_RAND_HEX = 8


def _generate(prefix: str, *, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Build a time-ordered identifier.

    Args:
        prefix: Short type prefix (for example ``"insp"``).
        now_ns: Injected wall-clock time in nanoseconds. Defaults to
            :func:`time.time_ns`. Exposed for deterministic testing.
        rand_hex: Injected random hex suffix. Defaults to a fresh UUID4
            fragment. Exposed for deterministic testing.

    Returns:
        The formatted identifier string.
    """
    nanos = time.time_ns() if now_ns is None else now_ns
    millis = nanos // 1_000_000
    suffix = uuid.uuid4().hex[:_RAND_HEX] if rand_hex is None else rand_hex
    return f"{prefix}-{millis:0{_MS_WIDTH}d}-{suffix}"


def new_inspection_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new inspection identifier (prefix ``insp``)."""
    return _generate("insp", now_ns=now_ns, rand_hex=rand_hex)


def new_part_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new part identifier (prefix ``part``)."""
    return _generate("part", now_ns=now_ns, rand_hex=rand_hex)


def new_frame_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new frame identifier (prefix ``frame``)."""
    return _generate("frame", now_ns=now_ns, rand_hex=rand_hex)


def new_trace_id(*, now_ns: int | None = None, rand_hex: str | None = None) -> str:
    """Return a new trace / correlation identifier (prefix ``trace``)."""
    return _generate("trace", now_ns=now_ns, rand_hex=rand_hex)
