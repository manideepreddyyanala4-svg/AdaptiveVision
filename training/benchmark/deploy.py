"""Deployment profiling: what each method actually costs on the line.

The lab card is not the production card. A benchmark that reports accuracy
alone will happily recommend a ViT-L memory bank for a station with a 40 ms
cycle time, and the recommendation will be wrong. This module measures the
numbers that decide deployability:

* **GPU latency** (p50/p95) at batch 1, which is how a station runs -- one
  part at a time, not a batch of 32.
* **CPU latency**, because many edge boxes have no GPU at all and the answer
  there is often a different model entirely.
* **Model size and peak VRAM**, which decide what fits on the box.
* **Throughput**, in parts per minute, which is the number a plant manager
  actually asks for.

p95 rather than mean: a station that misses its cycle time one frame in twenty
has a defect-escape problem, and a mean hides that entirely.

Usage:
    python training/benchmark/deploy.py --methods patchcore_wide_resnet50_2 dinomaly_vitb14
    python training/benchmark/deploy.py --from-leaderboard --dataset mvtec/bottle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):  # Allow `python training/benchmark/deploy.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark.methods_dinomaly
import benchmark.methods_native  # noqa: F401  (registers the native zoo)
from benchmark.data import load_split, parse_config
from benchmark.regimes import subsample_fit
from benchmark.registry import RunOptions, get

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "training" / "benchmark_results"

#: Forward passes discarded before timing, to let CUDA autotune kernels and
#: allocate its caches. Timing the first call measures warm-up, not the model.
_WARMUP_RUNS = 10

#: Timed passes. Enough for a stable p95 without dominating the sweep.
_TIMED_RUNS = 60


def _percentile_latency(durations: list[float]) -> dict[str, float]:
    """Summarize per-call durations (seconds) as latency and throughput."""
    array = np.asarray(durations) * 1000.0
    p50 = float(np.percentile(array, 50))
    p95 = float(np.percentile(array, 95))
    return {
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "latency_mean_ms": round(float(array.mean()), 3),
        # Parts per minute at the p95 latency, i.e. the rate the line can be
        # guaranteed rather than the rate it averages.
        "throughput_ppm_p95": round(60_000.0 / max(p95, 1e-6), 1),
    }


def measure_latency(
    scorer: torch.nn.Module, height: int, width: int, device: str
) -> dict[str, float]:
    """Time single-frame inference on one device.

    Args:
        scorer: A fitted scorer exposing ``embed`` and ``patch_scores``, or a
            model exposing ``anomaly_map``.
        height: Input height.
        width: Input width.
        device: Torch device string.

    Returns:
        Latency and throughput statistics.
    """
    torch_device = torch.device(device)
    scorer.to(torch_device).eval()
    if hasattr(scorer, "device"):
        scorer.device = torch_device

    frame = torch.rand(1, 3, height, width, device=torch_device) * 255.0
    forward = _forward_fn(scorer)

    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            forward(frame)
        _sync(torch_device)

        durations: list[float] = []
        for _ in range(_TIMED_RUNS):
            start = time.perf_counter()
            forward(frame)
            _sync(torch_device)
            durations.append(time.perf_counter() - start)

    return _percentile_latency(durations)


def _forward_fn(scorer: torch.nn.Module):
    """Return the callable that performs one scoring pass."""
    if hasattr(scorer, "model") and hasattr(scorer.model, "anomaly_map"):
        return lambda frame: scorer.model.anomaly_map(frame)
    return lambda frame: scorer.score_patches(scorer.embed(frame))


def _sync(device: torch.device) -> None:
    """Block until queued CUDA work finishes, so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _parameter_bytes(scorer: torch.nn.Module) -> int:
    """Total bytes held by parameters and buffers.

    Buffers matter more than parameters here: a PatchCore memory bank is
    entirely buffer, and it is what makes the artifact large.
    """
    total = sum(p.numel() * p.element_size() for p in scorer.parameters())
    total += sum(b.numel() * b.element_size() for b in scorer.buffers() if b is not None)
    return total


def profile_method(
    method_name: str,
    config_key: str,
    data_root: Path,
    options: RunOptions,
    devices: tuple[str, ...],
) -> dict[str, Any]:
    """Fit one method and profile its deployment characteristics.

    Returns:
        A profile row; ``status`` is ``"ok"`` or ``"failed"``.
    """
    row: dict[str, Any] = {"method": method_name, "config": config_key}
    try:
        spec = get(method_name)
        config = parse_config(config_key, data_root)
        train_paths, test_split = load_split(config, data_root)
        train_paths = subsample_fit(train_paths, options.max_fit_images, options.seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        scorer = spec.fit(config, train_paths, test_split, options)

        row.update(
            status="ok",
            exportable=spec.exportable,
            family=spec.family,
            height=config.height,
            width=config.width,
            model_mb=round(_parameter_bytes(scorer) / 1e6, 2),
            fit_peak_vram_gb=round(
                torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0, 3
            ),
        )

        for device in devices:
            if device.startswith("cuda") and not torch.cuda.is_available():
                continue
            stats = measure_latency(scorer, config.height, config.width, device)
            row.update({f"{device}_{key}": value for key, value in stats.items()})

    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _leaderboard_methods(ranking_csv: Path, limit: int) -> list[str]:
    """Read the top-ranked method names from the leaderboard's ranking table.

    Raises:
        SystemExit: If the ranking has not been generated yet.
    """
    import pandas as pd

    if not ranking_csv.exists():
        msg = f"No ranking at {ranking_csv}. Run training/benchmark/leaderboard.py first."
        raise SystemExit(msg)
    frame = pd.read_csv(ranking_csv)
    return [str(name) for name in frame["method"].head(limit)]


def main() -> None:
    """Parse arguments and profile the selected methods."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--from-leaderboard", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--ranking", type=Path, default=RESULTS_DIR / "ranking.csv")
    parser.add_argument("--dataset", default="mvtec/bottle", help="Config used for the fit.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--max-fit-images", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "deployment.jsonl")
    args = parser.parse_args()

    if args.from_leaderboard:
        methods = _leaderboard_methods(args.ranking, args.top)
    elif args.methods:
        methods = args.methods
    else:
        msg = "Pass --methods, or --from-leaderboard."
        raise SystemExit(msg)

    options = RunOptions(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_fit_images=args.max_fit_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        epochs=args.epochs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"profiling {len(methods)} methods on {args.dataset}\n")
    print(f"{'method':<38s} {'MB':>8s} {'gpu p95':>9s} {'cpu p95':>9s} {'ppm':>8s}")

    with args.output.open("w", encoding="utf-8") as handle:
        for method in methods:
            row = profile_method(method, args.dataset, args.data_root, options, tuple(args.devices))
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            if row["status"] == "ok":
                print(
                    f"{method:<38.38s} {row.get('model_mb', 0):>8.1f} "
                    f"{row.get('cuda_latency_p95_ms', float('nan')):>9.1f} "
                    f"{row.get('cpu_latency_p95_ms', float('nan')):>9.1f} "
                    f"{row.get('cuda_throughput_ppm_p95', float('nan')):>8.0f}"
                )
            else:
                print(f"{method:<38.38s} FAILED {row['error']}")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
