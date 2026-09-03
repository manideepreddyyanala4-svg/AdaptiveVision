"""Web API / dashboard entry point for AdaptiveVision (Milestone M15; M19 adds advisory/deployment).

This script wires the local SQLite result repository into the FastAPI
application at the composition root and serves the dashboard, REST API, and
Prometheus metrics over HTTP. Milestone M19's advisory repository (shares the
same database) and deployment profiles (an optional JSON artifact) are wired
in the same way - both degrade gracefully when absent, per M19's optional-
service principle.

Usage:
    python scripts/run_api.py

Then open http://127.0.0.1:8000/ in a browser to view the dashboard.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from adaptivevision.api import create_app
from adaptivevision.deployment import DeploymentProfile, load_deployment_profiles
from adaptivevision.storage import (
    SqliteAdvisoryRepository,
    SqliteResultRepository,
    open_database,
)

logger = logging.getLogger(__name__)

_DEFAULT_DEPLOYMENT_PROFILES = (
    Path(__file__).resolve().parent.parent
    / "training"
    / "benchmark_results"
    / "deployment_profiles.json"
)


def main() -> int:
    """Build the app and start the uvicorn server.

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(description="AdaptiveVision web API")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite database path (default: adaptivevision.db in cwd)",
    )
    parser.add_argument(
        "--deployment-profiles",
        type=Path,
        default=_DEFAULT_DEPLOYMENT_PROFILES,
        help=(
            "Path to a deployment_profiles.json artifact "
            "(written by training/benchmark/deployment_export.py). "
            "Missing is fine - the deployment routes just report no profiles."
        ),
    )
    args = parser.parse_args()

    _, session_factory = open_database(args.db)
    repository = SqliteResultRepository(session_factory)
    advisory = SqliteAdvisoryRepository(session_factory)

    deployment_profiles: tuple[DeploymentProfile, ...] = ()
    if args.deployment_profiles.exists():
        deployment_profiles = load_deployment_profiles(args.deployment_profiles)
    else:
        logger.info(
            "No deployment profiles at %s; /api/v1/deployment will report none.",
            args.deployment_profiles,
        )

    app = create_app(repository, advisory=advisory, deployment_profiles=deployment_profiles)

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
