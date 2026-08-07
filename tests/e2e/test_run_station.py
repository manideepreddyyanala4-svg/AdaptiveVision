"""End-to-end test for the M0 boot entrypoint (``scripts/run_station.py``).

This directly exercises the milestone acceptance criterion: booting emits one
structured log line that carries a populated ``correlation_id`` field.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_station.py"


def test_run_station_exits_cleanly_and_emits_boot_line() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected exactly one log line, got: {lines!r}"

    payload = json.loads(lines[-1])
    assert payload["message"] == "AdaptiveVision station starting"
    assert payload["level"] == "INFO"
    assert payload["correlation_id"].startswith("boot-")
    assert payload["state"] == "INIT"
    assert payload["milestone"] == "M0"
