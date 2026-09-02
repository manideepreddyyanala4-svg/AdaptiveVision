"""Run the entire study end to end, in one command.

Six stages, in dependency order:

1. **sweep** -- every method against every dataset configuration, in the
   one-class, multi-class and few-shot regimes, across every requested seed.
   This is the expensive stage and the only one that touches a GPU for long.
   Crash-safe and resumable at the individual-run level -- see run.py and
   store.py -- so relaunching this exact command after an interruption
   (including a reboot) picks up exactly where it stopped.
2. **cost** -- load each run's saved checkpoint and measure real inference
   latency/throughput/params/VRAM, backfilled onto the same row the sweep
   wrote. No re-fit.
3. **metrics** -- backfill PG2/PB2/AUPIMO from the sweep's stored ``.npz``
   artifacts, for any row that predates those metrics. No rerun.
4. **leaderboard** -- rank the sweep and write the Markdown report.
5. **ensemble** -- fuse methods from the stored artifacts. No GPU, seconds.
6. **deploy** -- profile latency/VRAM/size across CPU *and* GPU for the top
   methods (a separate, multi-device comparison from the cost stage above,
   which only measures the sweep's own device).

Each stage is skippable and each is independently resumable, so a run that
dies partway picks up where it stopped rather than starting over. Stages 2-6
are cheap and always re-run from whatever the sweep has produced so far.

Usage:
    python training/benchmark/run_all.py
    python training/benchmark/run_all.py --quick
    python training/benchmark/run_all.py --skip sweep
    python training/benchmark/run_all.py --dry-run
    python training/benchmark/run_all.py --only method=patchcore dataset=mvtec_loco
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_LOG = REPO_ROOT / "training" / "logs" / "run_all.log"

STAGES = ("sweep", "cost", "metrics", "leaderboard", "ensemble", "deploy")


def run_stage(name: str, command: list[str], required: bool) -> bool:
    """Run one stage as a subprocess.

    Stages run out-of-process so a segfault or CUDA OOM in one cannot take the
    orchestrator down with it -- the later stages still produce a report from
    whatever the sweep managed to finish.

    Args:
        name: Stage name, for logging.
        command: Argument list to execute.
        required: Whether a failure should abort the whole run.

    Returns:
        Whether the stage succeeded.

    Raises:
        SystemExit: If a required stage fails.
    """
    banner = f"  stage: {name}  "
    print(f"\n{'=' * 72}\n{banner:=^72}\n{'=' * 72}", flush=True)
    print("$ " + " ".join(command), flush=True)

    started = time.perf_counter()
    result = subprocess.run(command, check=False)
    elapsed = (time.perf_counter() - started) / 60

    if result.returncode == 0:
        print(f"\n[{name}] done in {elapsed:.1f} min", flush=True)
        return True

    print(f"\n[{name}] FAILED with exit code {result.returncode}", flush=True)
    if required:
        msg = f"Required stage {name!r} failed; stopping."
        raise SystemExit(msg)
    print(f"[{name}] not required -- continuing.", flush=True)
    return False


def main() -> None:
    """Parse arguments and run every requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--regimes", nargs="+", default=["oneclass", "multiclass"])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-fit-images", type=int, default=500)
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=2000,
        help="Caps Severstal's ~7,250-image split, which otherwise dominates runtime.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument(
        "--severstal-target-prevalence",
        type=float,
        default=None,
        help="Downsample Severstal's test split to this anomalous-image rate.",
    )
    parser.add_argument("--no-pixel", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip", nargs="+", default=[], choices=STAGES)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke run: 3 methods, 2 categories, tiny fit set, 1 seed. Minutes, not hours.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the sweep plan and ETA; run nothing at all."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Filter the sweep plan, e.g. --only method=patchcore dataset=mvtec_loco",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    results_db = results_dir / "benchmark.db"
    artifacts_dir = results_dir / "artifacts"
    checkpoints_dir = results_dir / "checkpoints"

    models = args.models
    datasets = args.datasets
    max_fit = args.max_fit_images
    max_test = args.max_test_images
    epochs = args.epochs
    seeds = args.seeds

    if args.quick:
        # A smoke test of the whole pipeline, including the parts that only
        # break on real data -- mask loading, artifact round-trips, the
        # gallery. Small enough to run before committing to the full sweep.
        models = ["patchcore_resnet18", "padim_wide_resnet50_2", "dfm_wide_resnet50_2"]
        datasets = ["mvtec/bottle", "mvtec/hazelnut"]
        max_fit, max_test = 60, 60
        epochs = epochs or 200
        seeds = seeds if seeds != [1, 2, 3] else [1]

    python = sys.executable
    started = time.perf_counter()
    completed: list[str] = []

    if "sweep" not in args.skip:
        command = [
            python,
            str(BENCHMARK_DIR / "run.py"),
            "--models",
            *models,
            "--datasets",
            *datasets,
            "--regimes",
            *args.regimes,
            "--data-root",
            str(args.data_root),
            "--results-db",
            str(results_db),
            "--artifacts",
            str(artifacts_dir),
            "--checkpoints",
            str(checkpoints_dir),
            "--log-file",
            str(DEFAULT_LOG),
            "--max-fit-images",
            str(max_fit),
            "--max-test-images",
            str(max_test),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seeds",
            *[str(seed) for seed in seeds],
            "--epochs",
            str(epochs),
        ]
        if args.severstal_target_prevalence is not None:
            command += ["--severstal-target-prevalence", str(args.severstal_target_prevalence)]
        if args.no_pixel:
            command.append("--no-pixel")
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
        if args.only:
            command += ["--only", *args.only]
        run_stage("sweep", command, required=True)
        completed.append("sweep")

    if args.dry_run:
        # A whole-pipeline dry-run shouldn't partially run downstream stages
        # against a stale/empty DB.
        return

    if "cost" not in args.skip and run_stage(
        "cost",
        [
            python,
            str(BENCHMARK_DIR / "cost.py"),
            "--results-db",
            str(results_db),
            "--checkpoints",
            str(checkpoints_dir),
        ],
        required=False,
    ):
        completed.append("cost")

    if "metrics" not in args.skip and run_stage(
        "metrics",
        [
            python,
            str(BENCHMARK_DIR / "metrics_backfill.py"),
            "--results-db",
            str(results_db),
            "--artifacts",
            str(artifacts_dir),
            "--data-root",
            str(args.data_root),
        ],
        required=False,
    ):
        completed.append("metrics")

    leaderboard_command = [
        python,
        str(BENCHMARK_DIR / "leaderboard.py"),
        "--results-db",
        str(results_db),
        "--output-dir",
        str(results_dir),
    ]
    if "leaderboard" not in args.skip and run_stage(
        "leaderboard", leaderboard_command, required=False
    ):
        completed.append("leaderboard")

    if "ensemble" not in args.skip:
        # Ensembling reads artifacts only, so a failure here costs nothing
        # that the rest of the report depends on.
        for regime in args.regimes:
            run_stage(
                f"ensemble[{regime}]",
                [
                    python,
                    str(BENCHMARK_DIR / "ensemble.py"),
                    "--results-db",
                    str(results_db),
                    "--artifacts",
                    str(artifacts_dir),
                    "--regime",
                    regime,
                    "--output",
                    str(results_dir / "ensembles.jsonl"),
                ],
                required=False,
            )
        completed.append("ensemble")

    if "deploy" not in args.skip:
        dataset_for_profile = datasets[0] if datasets[0] != "all" else "mvtec/bottle"
        if run_stage(
            "deploy",
            [
                python,
                str(BENCHMARK_DIR / "deploy.py"),
                "--from-leaderboard",
                "--ranking",
                str(results_dir / "ranking.csv"),
                "--dataset",
                dataset_for_profile,
                "--data-root",
                str(args.data_root),
                "--max-fit-images",
                str(min(200, max_fit)),
                "--output",
                str(results_dir / "deployment.jsonl"),
            ],
            required=False,
        ):
            completed.append("deploy")

    total = (time.perf_counter() - started) / 60
    print(f"\n{'=' * 72}")
    print(f"finished {len(completed)} stages in {total:.1f} min")
    print(f"  report:     {results_dir / 'leaderboard.md'}")
    print(f"  results db: {results_db}")
    print("=" * 72)


if __name__ == "__main__":
    main()
