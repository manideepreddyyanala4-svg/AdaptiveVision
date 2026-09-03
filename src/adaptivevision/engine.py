"""Analyze stage, part 1: run the model.

The ONNX Runtime inference engine and its latency/throughput profiling.
Profiling here measures the actual deployed :class:`~adaptivevision.common.InferenceEngine`
(for example an :class:`OnnxInferenceEngine` running under a specific
execution provider), so a production latency claim is measured against the
real deployment path rather than inferred from a pre-export PyTorch
measurement (that pre-export profiling lives in ``training/benchmark/cost.py``).
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adaptivevision.common import ExecutionProvider, InferenceEngine, InferenceError

_PROVIDER_NAMES: dict[ExecutionProvider, str] = {
    ExecutionProvider.CPU: "CPUExecutionProvider",
    ExecutionProvider.CUDA: "CUDAExecutionProvider",
    ExecutionProvider.TENSORRT: "TensorrtExecutionProvider",
    ExecutionProvider.OPENVINO: "OpenVINOExecutionProvider",
}


class OnnxInferenceEngine(InferenceEngine):
    """Inference engine backed by ONNX Runtime.

    Args:
        model_dir: Directory used to resolve relative model identifiers.
        providers: Preferred execution providers, in order.
        runtime: Optional ONNX Runtime-like module, used by tests.
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        *,
        providers: tuple[ExecutionProvider, ...] = (ExecutionProvider.CPU,),
        runtime: Any | None = None,
    ) -> None:
        """Initialize the engine without loading a model."""
        self._model_dir = Path(model_dir)
        self._providers = providers
        self._runtime = runtime
        self._session: Any | None = None
        self._model_version = ""

    @property
    def model_version(self) -> str:
        """Version identifier of the currently loaded model."""
        return self._model_version

    def load(self, model_id: str) -> None:
        """Load an ONNX model into an inference session."""
        model_path = self._resolve_model(model_id)
        runtime = self._runtime or _import_onnxruntime()
        try:
            self._session = runtime.InferenceSession(
                str(model_path),
                providers=[_PROVIDER_NAMES[p] for p in self._providers],
            )
        except Exception as exc:
            msg = f"Failed to load ONNX model {model_path}: {exc}"
            raise InferenceError(msg) from exc
        self._model_version = _session_model_version(self._session, model_path)

    def warmup(self) -> None:
        """Run a deterministic zero-input warmup when input shapes are static."""
        session = self._require_session()
        inputs = session.get_inputs()
        feed: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
        for input_meta in inputs:
            shape = tuple(
                1 if not isinstance(dim, int) or dim <= 0 else dim for dim in input_meta.shape
            )
            feed[input_meta.name] = np.zeros(shape, dtype=_numpy_dtype(input_meta.type))

        if feed:
            try:
                session.run(None, feed)
            except Exception as exc:
                msg = f"ONNX warmup failed: {exc}"
                raise InferenceError(msg) from exc

    def infer(
        self,
        inputs: Mapping[str, np.ndarray[Any, np.dtype[Any]]],
    ) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
        """Run inference on named input tensors."""
        session = self._require_session()
        try:
            output_values = session.run(None, dict(inputs))

        except Exception as exc:
            msg = f"ONNX inference failed: {exc}"
            raise InferenceError(msg) from exc
        output_names = [meta.name for meta in session.get_outputs()]
        return dict(zip(output_names, output_values, strict=True))

    def unload(self) -> None:
        """Unload the current model and clear version lineage."""
        self._session = None
        self._model_version = ""

    def _resolve_model(self, model_id: str) -> Path:
        """Resolve ``model_id`` to an existing ONNX model path."""
        path = Path(model_id)
        if not path.is_absolute():
            path = self._model_dir / path
        if not path.exists():
            msg = f"ONNX model not found: {path}"
            raise InferenceError(msg)
        return path

    def _require_session(self) -> Any:
        """Return the loaded session or raise an inference error."""
        if self._session is None:
            msg = "ONNX model is not loaded"
            raise InferenceError(msg)
        return self._session


def _import_onnxruntime() -> Any:
    """Import ONNX Runtime lazily with a domain-specific error."""
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        msg = "onnxruntime is not installed in the active environment"
        raise InferenceError(msg) from exc


def _session_model_version(session: Any, model_path: Path) -> str:
    """Extract model version metadata, falling back to the file stem."""
    try:
        metadata = session.get_modelmeta()
        custom = metadata.custom_metadata_map
        version = custom.get("version") or custom.get("model_version")
    except Exception:
        version = None
    return str(version or model_path.stem)


def _numpy_dtype(onnx_type: str) -> np.dtype[Any]:
    """Map ONNX Runtime tensor type strings to NumPy dtypes."""
    if "float16" in onnx_type:
        return np.dtype(np.float16)
    if "float" in onnx_type:
        return np.dtype(np.float32)
    if "double" in onnx_type:
        return np.dtype(np.float64)
    if "int64" in onnx_type:
        return np.dtype(np.int64)
    if "int32" in onnx_type:
        return np.dtype(np.int32)
    if "uint8" in onnx_type:
        return np.dtype(np.uint8)
    return np.dtype(np.float32)


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
    sample: Mapping[str, Any],
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
