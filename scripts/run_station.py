"""Boot entry point for the AdaptiveVision station (Milestone M3).

At M3 the station is a walking skeleton: it loads and validates its
configuration, configures structured logging, builds the full station through
the composition root (``adaptivevision.app``), boots it through the state
machine, runs a short demo inspection cycle against the null-object camera, and
shuts down cleanly.

Usage:
    python scripts/run_station.py

The log level and other settings are read from ``ADAPTIVEVISION_*`` environment
variables (see ``.env.example``).
"""

from __future__ import annotations

import uuid

from adaptivevision import __version__
from adaptivevision.app import build_station
from adaptivevision.config import load_config
from adaptivevision.logging_setup import (
    configure_logging,
    correlation_context,
    get_logger,
)


def main() -> int:
    """Boot the station, run a demo cycle, and shut down.

    Returns:
        Process exit code (``0`` on success).
    """
    config = load_config()
    configure_logging(level=config.log_level)

    logger = get_logger("adaptivevision.boot")
    boot_id = f"boot-{uuid.uuid4().hex[:12]}"

    with correlation_context(boot_id):
        station = build_station(config)
        station.boot()
        logger.info(
            "AdaptiveVision station booted",
            extra={
                "version": __version__,
                "milestone": "M3",
                "state": station.state.value,
                "station_id": config.station_id,
            },
        )

        station.ready()
        results = station.run(["demo-part-001"])
        for result in results:
            logger.info(
                "Inspection complete",
                extra={
                    "inspection_id": result.inspection_id,
                    "part_id": result.part_id,
                    "verdict": result.verdict.value,
                    "cycle_time_ms": result.cycle_time_ms,
                },
            )

        station.shutdown()
        logger.info(
            "AdaptiveVision station stopped",
            extra={
                "version": __version__,
                "milestone": "M3",
                "state": station.state.value,
                "station_id": config.station_id,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
