"""Sweep selected methods across selected datasets, in one or more regimes.

Results go to a SQLite store (``training/benchmark_results/benchmark.db``,
table ``runs``), one row per ``(method, regime, config, seed)``, keyed by a
deterministic ``run_id``. Before a fit starts, every row it will produce is
inserted as ``status="running"``; each is updated to ``"ok"``/``"failed"`` as
soon as that row is available, committed immediately. A row still
``"running"`` on the next launch is unambiguously a crash victim -- deleted
and retried automatically, no manual bookkeeping. A fitted model is also
saved to disk (``training/benchmark_results/checkpoints/``) before scoring,
for the deployment-cost pass (``cost.py``) to load and time later without
re-fitting.

Usage:
    python training/benchmark/run.py --list
    python training/benchmark/run.py --regimes oneclass multiclass --models all
    python training/benchmark/run.py --regimes multiclass --models dinomaly
    python training/benchmark/run.py --models patchcore --datasets mvtec
    python training/benchmark/run.py --dry-run
    python training/benchmark/run.py --only method=patchcore dataset=mvtec_loco
"""

from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):  # Allow `python training/benchmark/run.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark.methods_anomalib
import benchmark.methods_dinomaly
import benchmark.methods_native  # noqa: F401  (registers the native zoo)
from benchmark import store
from benchmark.data import DatasetConfig, discover_configs
from benchmark.logging_setup import configure_logging
from benchmark.planning import FitJob, build_run_plan
from benchmark.regimes import RunContext, run_fewshot, run_multiclass, run_oneclass
from benchmark.registry import RunOptions, all_methods, select
from benchmark.store import compute_run_id

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_RESULTS_DB = RESULTS_DIR / "benchmark.db"
DEFAULT_ARTIFACTS = RESULTS_DIR / "artifacts"
DEFAULT_CHECKPOINTS = RESULTS_DIR / "checkpoints"
DEFAULT_LOG = REPO_ROOT / "training" / "logs" / "run_all.log"

#: Regimes run when ``--regimes`` is not given. One-class is the literature
#: baseline; multi-class is the deployable result the report leads with.
DEFAULT_REGIMES = ("oneclass", "multiclass")

#: Rough per-job duration guesses, used for --dry-run's ETA only when this
#: DB has no historical timing for the method in question yet. Training-free
#: fits are seconds; gradient-trained (Dinomaly) fits are the real cost.
_FALLBACK_SECONDS_TRAINABLE = 45 * 60
_FALLBACK_SECONDS_TRAINING_FREE = 30


def select_configs(selectors: list[str], data_root: Path) -> list[DatasetConfig]:
    """Resolve ``--datasets`` selectors into concrete configurations.

    A selector is ``all``, a dataset family (``mvtec``), or a full key
    (``mvtec/bottle``).

    Raises:
        SystemExit: If a selector matches nothing on disk.
    """
    available = discover_configs(data_root)
    if not available:
        msg = f"No datasets found under {data_root}. Pass --data-root."
        raise SystemExit(msg)

    chosen: dict[str, DatasetConfig] = {}
    for selector in selectors:
        if selector == "all":
            chosen.update({config.key: config for config in available})
            continue
        matches = [c for c in available if c.key == selector or c.dataset == selector]
        if not matches:
            keys = ", ".join(sorted({c.dataset for c in available}))
            msg = f"Selector {selector!r} matched no dataset. Families present: {keys}"
            raise SystemExit(msg)
        chosen.update({config.key: config for config in matches})
    return sorted(chosen.values(), key=lambda config: config.key)


def _parse_only(tokens: list[str]) -> dict[str, set[str]]:
    """Parse ``--only key=value`` tokens into ``{key: {value, ...}}``.

    Same key repeated is OR'd together; different keys are AND'd.
    """
    parsed: dict[str, set[str]] = {}
    for token in tokens:
        if "=" not in token:
            msg = f"--only expects key=value tokens, got {token!r}"
            raise SystemExit(msg)
        key, value = token.split("=", 1)
        parsed.setdefault(key, set()).add(value)
    return parsed


def _job_matches_only(job: FitJob, only: dict[str, set[str]]) -> bool:
    """Whether ``job`` satisfies every ``--only`` constraint."""
    if "method" in only and not (
        job.spec.name in only["method"]
        or job.spec.family in only["method"]
        or job.spec.backend in only["method"]
    ):
        return False
    if "dataset" in only:
        configs = job.target if isinstance(job.target, list) else [job.target]
        if not any(c.dataset in only["dataset"] or c.key in only["dataset"] for c in configs):
            return False
    if "regime" in only and job.regime not in only["regime"]:
        return False
    return True


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch so a rerun reproduces the same row."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _describe(row: dict[str, Any]) -> str:
    """One-line summary of a finished run."""
    if row["status"] != "ok":
        return f"FAILED {row['error']}"
    parts = [
        f"AUROC={row['auroc']:.4f}",
        f"AP={row['average_precision']:.4f}",
        f"F1={row['f1_max']:.4f}",
        f"scrap@95={row['fpr_at_95tpr']:.3f}",
    ]
    if "aupro" in row and row["aupro"] == row["aupro"]:  # present and not NaN
        parts.append(f"AUPRO={row['aupro']:.4f}")
    parts.append(f"{row['ms_per_image']:.1f}ms/img")
    return " ".join(parts)


def job_label(job: FitJob) -> str:
    """Human-readable name for a unit of work."""
    if job.regime == "multiclass":
        return f"{job.spec.name} @ {job.target[0].dataset} (x{len(job.target)} categories) seed={job.seed}"
    return f"{job.spec.name} @ {job.target.key} seed={job.seed}"


def execute(job: FitJob, context: RunContext):
    """Dispatch one job to its regime runner."""
    if job.regime == "multiclass":
        return run_multiclass(job.spec, job.target, context)
    if job.regime == "fewshot":
        return run_fewshot(job.spec, job.target, context)
    return run_oneclass(job.spec, job.target, context)


def print_zoo() -> None:
    """Print the registered zoo, grouped by family."""
    methods = all_methods()
    print(f"{len(methods)} methods registered\n")
    family = None
    for spec in sorted(methods, key=lambda s: (s.family, s.name)):
        if spec.family != family:
            family = spec.family
            print(f"[{family}]")
        export = "onnx" if spec.exportable else "    "
        print(f"  {export}  {spec.name:38s} {spec.notes}")
    if not any(spec.backend == "anomalib" for spec in methods):
        print("\n(Anomalib not installed -- `pip install anomalib` adds its zoo.)")


def _estimate_seconds(
    session_factory: Any, pending: list[FitJob]
) -> float:
    """Blend real historical per-job durations with a rough fallback guess."""
    from sqlalchemy import select as sa_select

    from benchmark.store import RunRow, session_scope

    history: dict[str, list[float]] = {}
    with session_scope(session_factory) as session:
        rows = session.scalars(
            sa_select(RunRow).where(
                RunRow.status == "ok", RunRow.fit_seconds.is_not(None), RunRow.score_seconds.is_not(None)
            )
        )
        for row in rows:
            history.setdefault(row.method, []).append((row.fit_seconds or 0) + (row.score_seconds or 0))

    global_history = [seconds for seconds_list in history.values() for seconds in seconds_list]
    global_average = sum(global_history) / len(global_history) if global_history else None

    total = 0.0
    for job in pending:
        durations = history.get(job.spec.name)
        if durations:
            total += sum(durations) / len(durations)
        elif global_average is not None:
            total += global_average
        else:
            total += (
                _FALLBACK_SECONDS_TRAINABLE if job.spec.trainable else _FALLBACK_SECONDS_TRAINING_FREE
            )
    return total


def main() -> None:  # noqa: PLR0915 (the sweep loop is long by nature; splitting it hides the flow)
    """Parse arguments and run the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"], help="Names, families, or 'all'.")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Keys, families, or 'all'.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=list(DEFAULT_REGIMES),
        choices=["oneclass", "multiclass", "fewshot"],
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-fit-images", type=int, default=500, help="0 uses every image.")
    parser.add_argument("--max-test-images", type=int, default=0, help="0 uses the whole split.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs/iterations.")
    parser.add_argument(
        "--severstal-target-prevalence",
        type=float,
        default=None,
        help="Downsample Severstal's test split to this anomalous-image rate.",
    )
    parser.add_argument("--no-pixel", action="store_true", help="Skip localization metrics.")
    parser.add_argument("--force", action="store_true", help="Re-run cells that already passed.")
    parser.add_argument("--list", action="store_true", help="Print the zoo and exit.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and an ETA; run nothing."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Filter the plan, e.g. --only method=patchcore dataset=mvtec_loco",
    )
    args = parser.parse_args()

    if args.list:
        print_zoo()
        return

    logger = configure_logging(args.log_file)

    methods = select(args.models)
    configs = select_configs(args.datasets, args.data_root)
    plan = build_run_plan(
        methods,
        configs,
        args.regimes,
        seeds=tuple(args.seeds),
        severstal_target_prevalence=args.severstal_target_prevalence,
    )

    only = _parse_only(args.only)
    if only:
        plan = [job for job in plan if _job_matches_only(job, only)]

    _, session_factory = store.open_database(args.results_db)

    scope_ids = {result_spec.run_id for job in plan for result_spec in job.result_specs()}

    if not args.dry_run:
        deleted = store.reset_incomplete(session_factory, scope_ids, force=args.force)
        if deleted:
            logger.info("cleared %d incomplete/failed row(s) from a prior run", deleted)

    done_ids = set() if args.force else store.completed_run_ids(session_factory, scope_ids)
    pending = [job for job in plan if not all(rs.run_id in done_ids for rs in job.result_specs())]

    total_rows = len(scope_ids)
    only_note = f" (--only narrowed {len(methods)}x{len(configs)} selection)" if only else ""
    logger.info(
        "%d fits planned%s: %d regimes x %d seeds = %d rows total (%d already done)",
        len(plan),
        only_note,
        len(args.regimes),
        len(args.seeds),
        total_rows,
        len(done_ids),
    )
    logger.info("device=%s pixel_metrics=%s", args.device, not args.no_pixel)
    logger.info("results_db=%s", args.results_db)
    logger.info("host=%s torch=%s", platform.node(), torch.__version__)

    if args.dry_run:
        estimated_seconds = _estimate_seconds(session_factory, pending)
        logger.info(
            "\n--dry-run: %d fits pending, estimated %.1f min total. Nothing executed.",
            len(pending),
            estimated_seconds / 60,
        )
        return

    options = RunOptions(
        device=args.device,
        max_fit_images=args.max_fit_images,
        max_test_images=args.max_test_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seeds[0],
        epochs=args.epochs,
        severstal_target_prevalence=args.severstal_target_prevalence,
    )
    context = RunContext(
        args.data_root, options, args.artifacts, args.checkpoints, want_pixel=not args.no_pixel
    )

    started = time.perf_counter()
    elapsed_per_job: list[float] = []
    for index, job in enumerate(pending, start=1):
        options.seed = job.seed
        for result_spec in job.result_specs():
            store.start_run(session_factory, result_spec.run_id, job.identity_for(result_spec))

        remaining = len(pending) - index + 1
        if elapsed_per_job:
            avg = sum(elapsed_per_job) / len(elapsed_per_job)
            eta = f"ETA {avg * remaining / 60:.1f} min (avg {avg:.1f}s/fit)"
        else:
            eta = "ETA unknown (first fit)"
        logger.info("[%d/%d] %s: %s ... %s", index, len(pending), job.regime, job_label(job), eta)

        seed_everything(job.seed)
        job_start = time.perf_counter()
        for row in execute(job, context):
            row["elapsed_total_s"] = round(time.perf_counter() - started, 1)
            run_id = compute_run_id(
                row["method"],
                row["regime"],
                row["config"],
                row["seed"],
                row.get("defect_kind"),
                row.get("severstal_target_prevalence"),
            )
            store.finish_run(session_factory, run_id, row)
            logger.info("    %s: %s", row["config"], _describe(row))
        elapsed_per_job.append(time.perf_counter() - job_start)

    logger.info("\nDone in %.1f min -> %s", (time.perf_counter() - started) / 60, args.results_db)
    logger.info("Now run: python training/benchmark/leaderboard.py")


if __name__ == "__main__":
    main()
