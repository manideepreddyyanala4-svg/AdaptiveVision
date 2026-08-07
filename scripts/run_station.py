"""Minimal boot entry point for the AdaptiveVision station (Milestone M0).

At this milestone the station has no pipeline. This script exists to prove that
the packaging, logging, and tooling foundation is in place: it configures
structured logging, establishes a boot-scoped correlation id, emits a single
structured log line, and exits successfully.

Milestone M3 (Walking Skeleton) introduces the real composition root
(``adaptivevision.app``) and station controller; this script will then defer to
them. It deliberately implements no station behaviour now.

Usage:
    python scripts/run_station.py

The log level can be overridden with the ``ADAPTIVEVISION_LOG_LEVEL``
environment variable (full configuration loading arrives in Milestone M2).
"""

from __future__ import annotations

import os
import uuid

from adaptivevision import __version__
from adaptivevision.logging_setup import configure_logging, correlation_context, get_logger


def main() -> int:
    """Boot the station shell, emit one structured line, and exit.

    Returns:
        Process exit code (``0`` on success).
    """
    level = os.environ.get("ADAPTIVEVISION_LOG_LEVEL", "INFO")
    configure_logging(level=level)

    logger = get_logger("adaptivevision.boot")
    boot_id = f"boot-{uuid.uuid4().hex[:12]}"

    with correlation_context(boot_id):
        logger.info(
            "AdaptiveVision station starting",
            extra={"version": __version__, "milestone": "M0", "state": "INIT"},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
