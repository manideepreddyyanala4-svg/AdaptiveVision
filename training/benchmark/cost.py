"""Deployment-cost instrumentation, measured from a saved checkpoint.

This is the entire reason ``regimes.py`` saves a checkpoint for every run: a
cost pass that loads the fitted model back and times its forward pass,
instead of re-fitting just to get something to time (which is what
``deploy.py`` did before checkpoints existed, and still does for its own
separate multi-device CPU/GPU comparison -- this module is not a replacement
for that, it is what backfills ``inference_latency_ms_p50`` and friends onto
the same SQLite row the accuracy sweep already wrote).

Run as a stage after the sweep:

    python training/benchmark/cost.py --results-db training/benchmark_results/benchmark.db

It visits every ``status="ok"`` row missing cost columns, loads that run's
checkpoint, measures latency/throughput/params/VRAM, and writes the columns
back onto the same row via ``store.update_columns`` -- no re-fit, no re-score.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):  # Allow `python training/benchmark/cost.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import store
from benchmark.checkpoints import checkpoint_path, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"
DEFAULT_RESULTS_DB = RESULTS_DIR / "benchmark.db"
DEFAULT_CHECKPOINTS = RESULTS_DIR / "checkpoints"

#: Forward passes discarded before timing, to let CUDA autotune kernels and
#: allocate its caches.
_WARMUP_RUNS = 10

#: Timed batch-1 passes. The spec asks for >=100; comfortably clears it.
_TIMED_RUNS = 100


def _forward_fn(scorer: torch.nn.Module):
    """Return the callable that performs one scoring pass.

    Same dispatch as deploy.py's: Dinomaly exposes ``model.anomaly_map``;
    every native ``EmbeddingScorer`` (PatchCore/PaDiM/DFM) exposes
    ``embed``/``score_patches``.
    """
    if hasattr(scorer, "model") and hasattr(scorer.model, "anomaly_map"):
        return lambda frame: scorer.model.anomaly_map(frame)
    return lambda frame: scorer.score_patches(scorer.embed(frame))


def _sync(device: torch.device) -> None:
    """Block until queued CUDA work finishes, so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_batch(forward, frame: torch.Tensor, device: torch.device, runs: int) -> list[float]:
    """Warm up, then time ``runs`` forward passes. Returns per-call seconds."""
    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            forward(frame)
        _sync(device)

        durations: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            forward(frame)
            _sync(device)
            durations.append(time.perf_counter() - start)
    return durations


def measure_inference_cost(
    scorer: torch.nn.Module, height: int, width: int, device: str
) -> dict[str, float]:
    """Load-and-time a fitted scorer.

    Args:
        scorer: A checkpoint loaded via ``checkpoints.load_checkpoint``.
        height: Input height.
        width: Input width.
        device: Torch device string.

    Returns:
        ``inference_latency_ms_p50``, ``inference_latency_ms_p95``,
        ``throughput_fps_bs1``, ``throughput_fps_bs16`` (``nan`` if the
        method's forward path rejects a batch), ``model_params_millions``,
        ``peak_gpu_memory_mb`` (inference-only -- reset immediately before
        this function's own timed loop, distinct from the sweep's
        fit+score-spanning ``peak_vram_gb``).
    """
    torch_device = torch.device(device)
    scorer.to(torch_device).eval()
    if hasattr(scorer, "device"):
        scorer.device = torch_device

    # Some methods (Dinomaly) always run at their own fixed resolution
    # regardless of the dataset -- use that instead of the dataset's declared
    # geometry when the scorer exposes one, or timing would either crash
    # (input not a multiple of the patch size) or measure the wrong shape.
    height, width = getattr(scorer, "input_size", (height, width))

    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    forward = _forward_fn(scorer)

    frame_bs1 = torch.rand(1, 3, height, width, device=torch_device) * 255.0
    durations_bs1 = _time_batch(forward, frame_bs1, torch_device, _TIMED_RUNS)
    latency_ms = sorted(d * 1000.0 for d in durations_bs1)
    p50 = latency_ms[len(latency_ms) // 2]
    p95 = latency_ms[int(len(latency_ms) * 0.95)]
    mean_seconds_bs1 = sum(durations_bs1) / len(durations_bs1)

    try:
        frame_bs16 = torch.rand(16, 3, height, width, device=torch_device) * 255.0
        durations_bs16 = _time_batch(forward, frame_bs16, torch_device, 20)
        mean_seconds_bs16 = sum(durations_bs16) / len(durations_bs16)
        throughput_fps_bs16 = 16.0 / mean_seconds_bs16
    except RuntimeError:
        # The method's forward path doesn't support batching (or this batch
        # doesn't fit in VRAM) -- report it as unknown rather than failing
        # the whole cost measurement.
        throughput_fps_bs16 = float("nan")

    params_millions = sum(p.numel() for p in scorer.parameters()) / 1e6
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch_device.type == "cuda" else 0.0
    )

    return {
        "inference_latency_ms_p50": round(p50, 3),
        "inference_latency_ms_p95": round(p95, 3),
        "throughput_fps_bs1": round(1.0 / mean_seconds_bs1, 2),
        "throughput_fps_bs16": round(throughput_fps_bs16, 2)
        if throughput_fps_bs16 == throughput_fps_bs16  # not NaN
        else float("nan"),
        "model_params_millions": round(params_millions, 3),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 1),
    }


def _checkpoint_config_key(row) -> str:
    """The config key a row's checkpoint was saved under.

    Multi-class checkpoints are saved once per family (``row.dataset``, e.g.
    ``"mvtec"``), not per category -- every category's row shares that one
    fit. One-class and few-shot checkpoints are saved per the row's own
    ``config`` (few-shot's shot-specific ``regime`` string already
    disambiguates the path, same as at save time).
    """
    return row.dataset if row.regime == "multiclass" else row.config


def run_cost_pass(results_db: Path, checkpoint_root: Path, device: str) -> tuple[int, int]:
    """Visit every completed row missing cost data and backfill it.

    Returns:
        ``(updated, skipped_no_checkpoint)`` counts.
    """
    from sqlalchemy import select as sa_select

    from benchmark.store import RunRow, session_scope

    _, session_factory = store.open_database(results_db)

    with session_scope(session_factory) as session:
        rows = list(
            session.scalars(
                sa_select(RunRow).where(
                    RunRow.status == "ok", RunRow.inference_latency_ms_p50.is_(None)
                )
            )
        )
        # Copy out the plain values we need -- the ORM objects go stale once
        # this session closes, and update_columns opens its own session per row.
        pending = [
            (row.run_id, row.regime, row.method, _checkpoint_config_key(row), row.height, row.width)
            for row in rows
        ]

    updated = 0
    skipped = 0
    for index, (run_id, regime, method, config_key, height, width) in enumerate(pending, start=1):
        path = checkpoint_path(checkpoint_root, regime, method, config_key)
        scorer = load_checkpoint(path, device=device)
        if scorer is None:
            print(f"[{index}/{len(pending)}] {method} @ {config_key}: no checkpoint at {path}, skipping")
            skipped += 1
            continue
        cost = measure_inference_cost(scorer, height, width, device)
        store.update_columns(session_factory, run_id, cost)
        print(f"[{index}/{len(pending)}] {method} @ {config_key}: {cost}")
        updated += 1
        del scorer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return updated, skipped


def main() -> None:
    """Parse arguments and run the cost pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_RESULTS_DB)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    updated, skipped = run_cost_pass(args.results_db, args.checkpoints, args.device)
    print(f"\nbackfilled cost metrics on {updated} row(s), {skipped} skipped (no checkpoint found)")


if __name__ == "__main__":
    main()
