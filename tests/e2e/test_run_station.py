"""End-to-end test for the M3 boot entrypoint (``scripts/run_station.py``).

This directly exercises the milestone acceptance criterion: booting builds the
walking skeleton through the composition root, runs a demo inspection cycle
against the null-object camera, and shuts down cleanly, emitting structured log
lines throughout.

Deliberately runs with ``cwd`` pointed at an empty directory (via ``tmp_path``)
rather than the repo root: since M9/M10 were wired in (docs/milestones/M20.md),
a real ``.env`` with ``MODEL_PATH``/``DEFAULT_RECIPE_ID`` set (see
``.env.example``) makes the demo cycle score real inference and can legitimately
come back FAIL -- this test's job is only to prove the *skeleton* (no model
configured) still boots and runs cleanly, so it must not depend on whatever
``.env`` happens to be sitting in the working tree.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_station.py"


def test_run_station_exits_cleanly_and_emits_boot_line(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payloads = [json.loads(line) for line in lines]

    booted = next(p for p in payloads if p["message"] == "AdaptiveVision station booted")
    assert booted["level"] == "INFO"
    assert booted["correlation_id"].startswith("boot-")
    assert booted["state"] == "idle"
    assert booted["milestone"] == "M3"
    assert booted["station_id"] == "station-01"

    inspection = next(p for p in payloads if p["message"] == "Inspection complete")
    assert inspection["part_id"] == "demo-part-001"
    assert inspection["verdict"] == "pass"

    stopped = next(p for p in payloads if p["message"] == "AdaptiveVision station stopped")
    assert stopped["state"] == "shutdown"
    assert stopped["milestone"] == "M3"
