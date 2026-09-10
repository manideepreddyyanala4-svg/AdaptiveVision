"""Launch the AdaptiveVision fab dashboard and REST API on the edge IPC.

Starts the FastAPI application against the local SQLite traceability
database, serving:

- ``/``                     the operator dashboard (yield, Pareto, MES log)
- ``/metrics``              Prometheus text exposition for edge scraping
- ``/api/v1/results``       persisted inspection records
- ``/api/v1/advisory/...``  local-LLM advisory reports (when available)
- ``/api/v1/images/...``    frames the station archived for each part
- ``/api/v1/inspect-upload`` on-demand inspection of an uploaded sample

Everything runs locally: the database is a file on this machine and the
advisory engine talks to a local Ollama server. No cloud services are
contacted.

Usage:
    python run_dashboard.py

Environment:
    ADAPTIVEVISION_DB_PATH   SQLite file to serve (default ``adaptivevision.db``)
    ADAPTIVEVISION_API_HOST  Bind address (default ``127.0.0.1``)
    ADAPTIVEVISION_API_PORT  Bind port (default ``8000``)
    ADAPTIVEVISION_IMAGE_STORE_DIR
                             Frame archive to serve (default ``images``)
"""

from __future__ import annotations

import os

import uvicorn

from adaptivevision.api import create_app
from adaptivevision.app import build_manual_inspector
from adaptivevision.common import AdaptiveVisionError
from adaptivevision.config import load_config
from adaptivevision.storage import (
    LocalImageStore,
    SqliteAdvisoryRepository,
    SqliteResultRepository,
    build_engine,
    init_db,
    make_session_factory,
)

#: Default SQLite file backing the station's traceability records.
DEFAULT_DB_PATH = "adaptivevision.db"

#: Default bind address. Loopback by design: the dashboard is an on-machine
#: operator view, not a network service, unless a site explicitly rebinds it.
DEFAULT_HOST = "127.0.0.1"

#: Default bind port.
DEFAULT_PORT = 8000

#: Default archive directory the station writes inspected frames to. Must
#: match the station's own ``ADAPTIVEVISION_IMAGE_STORE_DIR``.
DEFAULT_IMAGE_DIR = "images"


def main() -> int:
    """Build the application and serve it until interrupted.

    Returns:
        Process exit code (``0`` on clean shutdown).
    """
    db_path = os.environ.get("ADAPTIVEVISION_DB_PATH", DEFAULT_DB_PATH)
    host = os.environ.get("ADAPTIVEVISION_API_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ADAPTIVEVISION_API_PORT", str(DEFAULT_PORT)))
    image_dir = os.environ.get("ADAPTIVEVISION_IMAGE_STORE_DIR", DEFAULT_IMAGE_DIR)

    engine = build_engine(f"sqlite:///{db_path}")
    init_db(engine)
    session_factory = make_session_factory(engine)

    repository = SqliteResultRepository(session_factory)
    advisory = SqliteAdvisoryRepository(session_factory)
    image_store = LocalImageStore(image_dir)

    # Teach mode needs a real pipeline. Building it can fail for ordinary
    # reasons (no model configured, an unreadable artifact); the dashboard
    # still serves its read-only views in that case, and simply does not
    # offer the ingestion panel.
    manual_inspector = None
    try:
        manual_inspector, _, _ = build_manual_inspector(load_config())
    except AdaptiveVisionError as exc:
        print(f"Teach mode unavailable: {exc.message}")

    app = create_app(
        repository,
        advisory=advisory,
        image_store=image_store,
        manual_inspector=manual_inspector,
    )

    print(f"AdaptiveVision dashboard: http://{host}:{port}")
    print(f"Prometheus metrics:       http://{host}:{port}/metrics")
    print(f"Traceability database:    {db_path}")
    print(f"Image archive:            {image_dir}")
    print(f"Teach mode:               {'enabled' if manual_inspector else 'unavailable'}")

    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
