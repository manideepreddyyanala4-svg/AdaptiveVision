"""Unit tests for engine.py: ONNX inference and latency profiling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from adaptivevision.common import ExecutionProvider, InferenceEngine, InferenceError
from adaptivevision.engine import OnnxInferenceEngine, benchmark_latency

# -----------------------------------------------------------------------------
# ONNX inference engine
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class _Meta:
    name: str
    shape: tuple[int, ...]
    type: str


class _ModelMeta:
    custom_metadata_map: ClassVar[dict[str, str]] = {"version": "model-v1"}


class _Session:
    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path
        self.providers = providers
        self.runs: list[dict[str, np.ndarray[Any, np.dtype[Any]]]] = []

    def get_modelmeta(self) -> _ModelMeta:
        return _ModelMeta()

    def get_inputs(self) -> list[_Meta]:
        return [_Meta("input", (1, 2), "tensor(float)")]

    def get_outputs(self) -> list[_Meta]:
        return [_Meta("output", (1, 2), "tensor(float)")]

    def run(
        self,
        output_names: object,
        inputs: dict[str, np.ndarray[Any, np.dtype[Any]]],
    ) -> list[np.ndarray[Any, np.dtype[Any]]]:
        _ = output_names
        self.runs.append(inputs)
        return [inputs["input"] + 1.0]


class _Runtime:
    def __init__(self) -> None:
        self.session: _Session | None = None

    def InferenceSession(  # noqa: N802
        self, path: str, providers: list[str]
    ) -> _Session:
        self.session = _Session(path, providers)
        return self.session


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx")
    return path


def test_onnx_engine_loads_model_and_records_version(tmp_path) -> None:
    runtime = _Runtime()
    model = _model_file(tmp_path)
    engine = OnnxInferenceEngine(
        model_dir=tmp_path,
        providers=(ExecutionProvider.CPU,),
        runtime=runtime,
    )
    engine.load(model.name)
    assert engine.model_version == "model-v1"
    assert runtime.session is not None
    assert runtime.session.providers == ["CPUExecutionProvider"]


def test_onnx_engine_warmup_and_infer(tmp_path) -> None:
    runtime = _Runtime()
    engine = OnnxInferenceEngine(model_dir=tmp_path, runtime=runtime)
    engine.load(_model_file(tmp_path).name)
    engine.warmup()
    output = engine.infer({"input": np.array([[1.0, 2.0]], dtype=np.float32)})
    np.testing.assert_array_equal(
        output["output"], np.array([[2.0, 3.0]], dtype=np.float32)
    )


def test_onnx_engine_rejects_missing_model(tmp_path) -> None:
    engine = OnnxInferenceEngine(model_dir=tmp_path, runtime=_Runtime())
    with pytest.raises(InferenceError, match="not found"):
        engine.load("missing.onnx")


def test_onnx_engine_requires_loaded_model(tmp_path) -> None:
    engine = OnnxInferenceEngine(model_dir=tmp_path, runtime=_Runtime())
    with pytest.raises(InferenceError, match="not loaded"):
        engine.infer({"input": np.array([[1.0]], dtype=np.float32)})


def test_onnx_engine_unload_clears_version(tmp_path) -> None:
    engine = OnnxInferenceEngine(model_dir=tmp_path, runtime=_Runtime())
    engine.load(_model_file(tmp_path).name)
    engine.unload()
    assert engine.model_version == ""


# -----------------------------------------------------------------------------
# Latency/throughput profiling
# -----------------------------------------------------------------------------

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
