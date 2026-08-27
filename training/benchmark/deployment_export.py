"""Export validated DeploymentProfile records from the completed sweep (Milestone M19).

This is the one-way bridge between the research sweep and production: reads
the sweep's SQLite store (already populated with accuracy *and* cost metrics
by ``leaderboard.py``'s columns and ``cost.py``'s backfill pass) and writes a
small, versioned JSON artifact. ``adaptivevision.deployment.profiles``
(production side, under ``src/``) only ever reads that JSON file - it has no
dependency on this module, on training-only packages (torch, pandas), or on
the sweep database directly, so production never has to trust an in-progress
or unvalidated run.

Usage:
    python training/benchmark/deployment_export.py
    python training/benchmark/deployment_export.py --results-db path/to/benchmark.db
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # Allow `python training/benchmark/deployment_export.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from benchmark.leaderboard import aggregate_seeds, load_results

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DB = REPO_ROOT / "training" / "benchmark_results" / "benchmark.db"
DEFAULT_OUTPUT = REPO_ROOT / "training" / "benchmark_results" / "deployment_profiles.json"

#: One profile per deployable model configuration per dataset - "category" is
#: a defect label within a dataset, not a separate deployable model, so it is
#: intentionally excluded from the grouping (unlike ``leaderboard.py``'s
#: default, which is per-regime rather than per-dataset).
GROUP_COLS: tuple[str, ...] = ("method", "family", "backend", "config", "dataset")

#: Aggregated (``..._mean``) source column -> DeploymentProfile field name.
_PROFILE_FIELDS: dict[str, str] = {
    "auroc_mean": "image_auroc",
    "pixel_auroc_mean": "pixel_auroc",
    "inference_latency_ms_p50_mean": "p50_latency_ms",
    "inference_latency_ms_p95_mean": "p95_latency_ms",
    "throughput_fps_bs1_mean": "throughput_fps",
    "model_params_millions_mean": "model_params_millions",
    "peak_gpu_memory_mb_mean": "peak_gpu_memory_mb",
    "training_wall_clock_seconds_mean": "training_wall_clock_seconds",
}


def build_profiles(frame: pd.DataFrame, *, benchmark_version: str) -> list[dict[str, Any]]:
    """Aggregate seeds and shape one DeploymentProfile dict per model config.

    Args:
        frame: The raw per-run frame, as returned by
            :func:`benchmark.leaderboard.load_results`.
        benchmark_version: Free-form label identifying this sweep (for
            example a git SHA or a sweep date), recorded on every profile.

    Returns:
        One JSON-serializable dict per ``(method, family, backend, config,
        dataset)``, with ``None`` for any metric that has no data.
    """
    agg = aggregate_seeds(frame, group_cols=GROUP_COLS)
    validated_at = datetime.now(UTC).isoformat()
    profiles: list[dict[str, Any]] = []
    for _, row in agg.iterrows():
        profile: dict[str, Any] = {
            "model": str(row["method"]),
            "family": str(row["family"]),
            "backbone": str(row["backend"]),
            "config": str(row["config"]),
            "dataset": str(row["dataset"]),
            "n_seeds": int(row["n_seeds"]),
            "benchmark_version": benchmark_version,
            "validated_at": validated_at,
        }
        for src_col, dst_field in _PROFILE_FIELDS.items():
            value = row.get(src_col)
            profile[dst_field] = None if value is None or pd.isna(value) else float(value)
        profiles.append(profile)
    return profiles


def export_deployment_profiles(
    results_db: Path, output_path: Path, *, benchmark_version: str
) -> int:
    """Read ``results_db`` and write ``output_path``.

    Returns:
        The number of profiles written.
    """
    frame = load_results(results_db)
    profiles = build_profiles(frame, benchmark_version=benchmark_version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    return len(profiles)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--benchmark-version",
        default=datetime.now(UTC).strftime("%Y%m%d"),
        help="Free-form label recorded on every profile (default: today's date).",
    )
    args = parser.parse_args()

    n = export_deployment_profiles(
        args.results_db, args.output, benchmark_version=args.benchmark_version
    )
    print(f"wrote {n} deployment profiles to {args.output}")


if __name__ == "__main__":
    main()
