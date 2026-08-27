"""Unit tests for M8 ONNX inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from adaptivevision.common.enums import ExecutionProvider
from adaptivevision.common.errors import InferenceError
from adaptivevision.inference import OnnxInferenceEngine


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
