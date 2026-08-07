"""Shared pytest fixtures for the AdaptiveVision test suite.

Logging is global process state, so these fixtures snapshot and restore it
around every test to keep tests independent and order-insensitive.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest

from adaptivevision import logging_setup


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Snapshot and restore root logger handlers/level around each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


@pytest.fixture(autouse=True)
def _reset_correlation_id() -> Iterator[None]:
    """Ensure each test starts and ends with the default correlation id."""
    logging_setup.clear_correlation_id()
    try:
        yield
    finally:
        logging_setup.clear_correlation_id()


@pytest.fixture
def log_stream() -> io.StringIO:
    """Configure logging to write JSON lines into an in-memory buffer."""
    stream = io.StringIO()
    logging_setup.configure_logging(level="DEBUG", stream=stream)
    return stream
