"""Durable progress logging for an unattended multi-hour sweep.

``run.py`` already prints progress to stdout, which is enough for a terminal
someone is watching. This adds the same lines to a rotating file too, so
progress since the last reboot/relaunch is inspectable without having kept a
terminal open (``tail -f training/logs/run_all.log``).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "benchmark"


def configure_logging(log_path: Path) -> logging.Logger:
    """Attach a rotating file handler (and, if absent, a stdout handler).

    Idempotent -- safe to call from both ``run_all.py`` and ``run.py`` when
    one launches the other as a subprocess pointed at the same file.

    Args:
        log_path: Destination log file. Created, along with its parent
            directory, if absent.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path.resolve())
        for h in logger.handlers
    )
    if not has_file_handler:
        file_handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(file_handler)

    # RotatingFileHandler is itself a StreamHandler subclass, so this must
    # exclude file handlers explicitly or it never adds the console one.
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger
