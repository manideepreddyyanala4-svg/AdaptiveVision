"""Unit tests for the M19 FAISS retrieval index."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adaptivevision.common import RetrievalError
from adaptivevision.explanation import FaissRetrievalIndex


class _FakeIndex:
    """A brute-force stand-in for a FAISS flat index."""

    def __init__(self, dim: int, *, use_inner_product: bool, fail: bool = False) -> None:
        self.dim = dim
        self.use_inner_product = use_inner_product
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.fail = fail

    def add(self, vectors: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        if self.fail:
            msg = "simulated FAISS add failure"
            raise RuntimeError(msg)
        self.vectors = np.vstack([self.vectors, vectors])

    def search(
        self, query: np.ndarray[Any, np.dtype[np.float32]], k: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self.fail:
            msg = "simulated FAISS search failure"
            raise RuntimeError(msg)
        n = self.vectors.shape[0]
        if n == 0:
            return (
                np.full((1, k), -1.0, dtype=np.float32),
                np.full((1, k), -1, dtype=np.int64),
            )
        if self.use_inner_product:
            scores = self.vectors @ query[0]
            order = np.argsort(-scores)
        else:
            scores = np.linalg.norm(self.vectors - query[0], axis=1)
            order = np.argsort(scores)
        top = order[:k]
        padded_idx = np.full(k, -1, dtype=np.int64)
        padded_dist = np.full(k, -1.0, dtype=np.float32)
        padded_idx[: len(top)] = top
        padded_dist[: len(top)] = scores[top]
        return padded_dist.reshape(1, -1), padded_idx.reshape(1, -1)


class _FakeFaiss:
    """A minimal FAISS-module stand-in, used by tests."""

    def __init__(self, *, fail_index: bool = False, fail_io: bool = False) -> None:
        self._fail_index = fail_index
        self._fail_io = fail_io

    def IndexFlatL2(self, dim: int) -> _FakeIndex:  # noqa: N802
        return _FakeIndex(dim, use_inner_product=False, fail=self._fail_index)

    def IndexFlatIP(self, dim: int) -> _FakeIndex:  # noqa: N802
        return _FakeIndex(dim, use_inner_product=True, fail=self._fail_index)

    def write_index(self, index: _FakeIndex, path: str) -> None:
        if self._fail_io:
            msg = "simulated FAISS write_index failure"
            raise RuntimeError(msg)
        Path(path).write_bytes(pickle.dumps(index))

    def read_index(self, path: str) -> _FakeIndex:
        if self._fail_io:
            msg = "simulated FAISS read_index failure"
            raise RuntimeError(msg)
        return pickle.loads(Path(path).read_bytes())


def _index(**kwargs: Any) -> FaissRetrievalIndex:
    return FaissRetrievalIndex(4, faiss_module=_FakeFaiss(), **kwargs)


def _rows(*values: list[float]) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.array(values, dtype=np.float32)


def _meta(n: int) -> list[dict[str, str]]:
    return [
        {"dataset": "mvtec_ad", "category": "bottle", "defect_type": f"defect-{i}"}
        for i in range(n)
    ]


def test_add_assigns_sequential_ids_and_returns_them() -> None:
    index = _index()
    ids = index.add(_rows([1, 0, 0, 0], [0, 1, 0, 0]), _meta(2))
    assert ids == (0, 1)
    more_ids = index.add(_rows([0, 0, 1, 0]), _meta(1))
    assert more_ids == (2,)


def test_search_returns_nearest_match_with_metadata() -> None:
    index = _index(metric="l2")
    index.add(
        _rows([0, 0, 0, 0], [10, 10, 10, 10], [1, 0, 0, 0]),
        [
            {"dataset": "mvtec_ad", "category": "bottle", "defect_type": "crack"},
            {"dataset": "visa", "category": "capsules", "defect_type": "poke"},
            {"dataset": "kolektorsdd2", "category": "surface", "defect_type": "blob"},
        ],
    )
    results = index.search(np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].vector_id == 0
    assert results[0].dataset == "mvtec_ad"
    assert results[0].defect_type == "crack"


def test_search_omits_padding_when_fewer_than_top_k_results() -> None:
    index = _index()
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    results = index.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5)
    assert len(results) == 1


def test_cosine_metric_normalizes_before_add_and_search() -> None:
    index = _index(metric="cosine")
    # Two vectors that are positive scalar multiples of each other should be
    # indistinguishable after L2 normalization.
    index.add(_rows([1, 0, 0, 0], [50, 0, 0, 0]), _meta(2))
    results = index.search(np.array([2, 0, 0, 0], dtype=np.float32), top_k=2)
    assert {r.vector_id for r in results} == {0, 1}
    assert results[0].distance == pytest.approx(results[1].distance, abs=1e-5)


def test_add_rejects_dimension_mismatch() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match="dimension"):
        index.add(_rows([1, 2, 3]), _meta(1))


def test_add_rejects_non_finite_values() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match=r"NaN|infinite"):
        index.add(_rows([1, float("nan"), 0, 0]), _meta(1))


def test_add_rejects_metadata_length_mismatch() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match="metadata"):
        index.add(_rows([1, 0, 0, 0], [0, 1, 0, 0]), _meta(1))


def test_add_accepts_empty_input() -> None:
    index = _index()
    assert index.add(np.empty((0, 4), dtype=np.float32), []) == ()


def test_search_rejects_dimension_mismatch() -> None:
    index = _index()
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    with pytest.raises(RetrievalError, match="dimension"):
        index.search(np.array([1, 0, 0], dtype=np.float32))


def test_unsupported_index_type_rejected() -> None:
    with pytest.raises(RetrievalError, match="index_type"):
        FaissRetrievalIndex(4, index_type="ivf", faiss_module=_FakeFaiss())


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = _index(embedding_model="dinov2", embedding_version="v1")
    index.add(_rows([1, 0, 0, 0], [0, 1, 0, 0]), _meta(2))
    path = tmp_path / "index.faiss"
    index.save(path)

    reloaded = FaissRetrievalIndex(
        4, faiss_module=_FakeFaiss(), embedding_model="dinov2", embedding_version="v1"
    )
    reloaded.load(path)
    results = reloaded.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
    assert results[0].vector_id == 0
    assert results[0].dataset == "mvtec_ad"


def test_load_rejects_incompatible_embedding_configuration(tmp_path: Path) -> None:
    index = _index(embedding_model="dinov2", embedding_version="v1")
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    path = tmp_path / "index.faiss"
    index.save(path)

    mismatched = FaissRetrievalIndex(
        4, faiss_module=_FakeFaiss(), embedding_model="resnet50", embedding_version="v1"
    )
    with pytest.raises(RetrievalError, match="incompatible"):
        mismatched.load(path)


def test_import_faiss_raises_retrieval_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from adaptivevision import explanation as faiss_index

    real_import_module = importlib.import_module

    def _raise(name: str, *args: object, **kwargs: object) -> object:
        if name == "faiss":
            raise ImportError(name)
        return real_import_module(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(faiss_index.importlib, "import_module", _raise)
    with pytest.raises(RetrievalError, match="faiss is not installed"):
        faiss_index._import_faiss()


def test_metadata_info_exposes_index_configuration() -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(), metric="l2", embedding_model="dinov2")
    info = index.metadata_info
    assert info.dim == 4
    assert info.metric == "l2"
    assert info.embedding_model == "dinov2"


def test_add_wraps_underlying_faiss_failure() -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(fail_index=True))
    with pytest.raises(RetrievalError, match="Failed to add"):
        index.add(_rows([1, 0, 0, 0]), _meta(1))


def test_search_wraps_underlying_faiss_failure() -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(fail_index=True))
    with pytest.raises(RetrievalError, match="FAISS search failed"):
        index.search(np.array([1, 0, 0, 0], dtype=np.float32))


def test_save_wraps_underlying_io_failure(tmp_path: Path) -> None:
    index = FaissRetrievalIndex(4, faiss_module=_FakeFaiss(fail_io=True))
    with pytest.raises(RetrievalError, match="Failed to save"):
        index.save(tmp_path / "index.faiss")


def test_load_wraps_generic_failure_distinctly_from_incompatibility(
    tmp_path: Path,
) -> None:
    # A missing/corrupt sidecar is a different failure mode than the
    # "incompatible configuration" RetrievalError raised deliberately -
    # both must surface as RetrievalError, but via the generic except path.
    index = _index()
    with pytest.raises(RetrievalError, match="Failed to load"):
        index.load(tmp_path / "does-not-exist.faiss")


def test_add_rejects_wrong_number_of_dimensions() -> None:
    index = _index()
    with pytest.raises(RetrievalError, match="Expected a 2D array"):
        index.add(np.array([1, 0, 0, 0], dtype=np.float32), _meta(1))


def test_search_rejects_wrong_number_of_dimensions() -> None:
    index = _index()
    index.add(_rows([1, 0, 0, 0]), _meta(1))
    with pytest.raises(RetrievalError, match="Expected a 1D array"):
        index.search(np.array([[1, 0, 0, 0]], dtype=np.float32).reshape(1, 1, 4))
