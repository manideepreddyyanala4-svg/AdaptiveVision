"""ONNX Runtime inference engine (Milestone M8)."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from adaptivevision.common.enums import ExecutionProvider
from adaptivevision.common.errors import InferenceError
from adaptivevision.common.interfaces import InferenceEngine

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
                1 if not isinstance(dim, int) or dim <= 0 else dim
                for dim in input_meta.shape
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
