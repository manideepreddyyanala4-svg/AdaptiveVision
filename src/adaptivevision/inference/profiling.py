"""Latency/throughput profiling for a production inference engine (Milestone M19).

Complements ``training/benchmark/cost.py``, which profiles a fitted PyTorch
scorer *before* export. This module profiles the actual deployed
:class:`~adaptivevision.common.interfaces.InferenceEngine` (for example an
``OnnxInferenceEngine`` running under a specific execution provider), so a
production latency claim is measured against the real deployment path rather
than inferred from a pre-export PyTorch measurement.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from adaptivevision.common.interfaces import InferenceEngine
    from adaptivevision.common.types import Image


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """Result of a :func:`benchmark_latency` run.

    Attributes:
        p50_latency_ms: Median single-inference latency.
        p95_latency_ms: 95th-percentile single-inference latency.
        throughput_fps: Single-inference throughput (``1 / mean latency``).
        model_version: The engine's loaded model version at measurement time.
        warmup: Warmup iterations run and discarded before timing.
        iters: Timed iterations.
        peak_gpu_memory_mb: Peak GPU memory allocated during the timed loop,
            or ``None`` if it could not be measured (no CUDA-capable torch
            installed, or CUDA unavailable) - never a fabricated number.
    """

    p50_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float
    model_version: str
    warmup: int
    iters: int
    peak_gpu_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_fps": self.throughput_fps,
            "model_version": self.model_version,
            "warmup": self.warmup,
            "iters": self.iters,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
        }


def benchmark_latency(
    engine: InferenceEngine,
    sample: Mapping[str, Image],
    *,
    warmup: int = 20,
    iters: int = 100,
) -> LatencyProfile:
    """Measure single-inference latency/throughput of an already-loaded engine.

    Args:
        engine: An inference engine with ``load()`` already called.
        sample: Input tensors matching what ``engine.infer`` expects.
        warmup: Iterations run and discarded before timing starts.
        iters: Timed iterations.

    Returns:
        The measured latency profile.

    Raises:
        InferenceError: Propagated from ``engine.infer`` on failure - this
            function performs no error translation of its own.
    """
    for _ in range(warmup):
        engine.infer(sample)

    _reset_peak_gpu_memory()
    durations_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        engine.infer(sample)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    durations_ms.sort()
    p50 = durations_ms[len(durations_ms) // 2]
    p95 = durations_ms[min(int(len(durations_ms) * 0.95), len(durations_ms) - 1)]
    mean_seconds = (sum(durations_ms) / len(durations_ms)) / 1000.0
    throughput = round(1.0 / mean_seconds, 2) if mean_seconds > 0 else float("inf")

    return LatencyProfile(
        p50_latency_ms=round(p50, 4),
        p95_latency_ms=round(p95, 4),
        throughput_fps=throughput,
        model_version=engine.model_version,
        warmup=warmup,
        iters=iters,
        peak_gpu_memory_mb=_peak_gpu_memory_mb(),
    )


def _reset_peak_gpu_memory() -> None:
    """Reset CUDA peak-memory stats if torch+CUDA are available, else no-op."""
    torch = _try_import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float | None:
    """Return peak CUDA memory in MB since the last reset, or ``None``."""
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / 1e6


def _try_import_torch() -> Any | None:
    """Import torch lazily, returning ``None`` if unavailable.

    ``src/`` has no hard dependency on torch (it is training-only, per
    ``training/requirements.txt``); GPU memory profiling degrades to
    ``None`` rather than failing when it is absent.
    """
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None
