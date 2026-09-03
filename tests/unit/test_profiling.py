"""Unit tests for the M19 inference latency profiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from adaptivevision.common import InferenceEngine
from adaptivevision.engine import benchmark_latency


class _FakeEngine(InferenceEngine):
    """A trivial engine that records how many times it was called."""

    def __init__(self, *, version: str = "fake-v1") -> None:
        self.calls = 0
        self._version = version

    @property
    def model_version(self) -> str:
        return self._version

    def load(self, model_id: str) -> None:
        pass

    def warmup(self) -> None:
        pass

    def infer(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        return {"output": inputs["input"]}

    def unload(self) -> None:
        pass


def test_benchmark_latency_runs_warmup_plus_iters_calls() -> None:
    engine = _FakeEngine()
    sample = {"input": np.zeros((1, 3, 8, 8), dtype=np.float32)}
    benchmark_latency(engine, sample, warmup=5, iters=10)
    assert engine.calls == 15


def test_benchmark_latency_reports_expected_fields() -> None:
    engine = _FakeEngine(version="patchcore-v3")
    sample = {"input": np.zeros((1, 3, 8, 8), dtype=np.float32)}
    profile = benchmark_latency(engine, sample, warmup=2, iters=20)

    assert profile.model_version == "patchcore-v3"
    assert profile.warmup == 2
    assert profile.iters == 20
    assert profile.p50_latency_ms >= 0.0
    assert profile.p95_latency_ms >= profile.p50_latency_ms
    assert profile.throughput_fps > 0.0
    # No CUDA-capable torch is assumed present in the test environment.
    assert profile.peak_gpu_memory_mb is None or profile.peak_gpu_memory_mb >= 0.0


def test_benchmark_latency_to_dict_round_trips_keys() -> None:
    engine = _FakeEngine()
    sample = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    profile = benchmark_latency(engine, sample, warmup=1, iters=5)
    data = profile.to_dict()
    assert set(data) == {
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_fps",
        "model_version",
        "warmup",
        "iters",
        "peak_gpu_memory_mb",
    }


class _FakeCuda:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.peak_bytes = 123_456_789

    def is_available(self) -> bool:
        return True

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def max_memory_allocated(self) -> int:
        return self.peak_bytes


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


def test_benchmark_latency_reports_gpu_memory_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptivevision import engine as profiling

    fake_torch = _FakeTorch()
    monkeypatch.setattr(profiling, "_try_import_torch", lambda: fake_torch)

    engine = _FakeEngine()
    sample = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    profile = benchmark_latency(engine, sample, warmup=1, iters=1)

    assert fake_torch.cuda.reset_calls == 1
    assert profile.peak_gpu_memory_mb == pytest.approx(123_456_789 / 1e6)


def test_benchmark_latency_propagates_infer_errors() -> None:
    class _FailingEngine(_FakeEngine):
        def infer(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        benchmark_latency(_FailingEngine(), {"input": np.zeros((1, 1))}, warmup=1, iters=1)
