"""FAISS-backed historical-defect retrieval index (Milestone M19).

FAISS stores and searches vectors only; it is never the source of truth for
business metadata. Each index keeps a small, versioned JSON sidecar
(:class:`IndexMetadata`) alongside the binary FAISS file so a saved index
records the embedding model/version and preprocessing version it was built
with, and refuses to silently mix incompatible embeddings back in on load.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np

from adaptivevision.common.errors import RetrievalError
from adaptivevision.common.interfaces import RetrievalIndex
from adaptivevision.common.result import RetrievalMatch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from adaptivevision.common.types import Embedding

#: FAISS index flavors implemented at Milestone M19. ``index_type`` is a
#: configuration point for future milestones; only "flat" is supported today.
_SUPPORTED_INDEX_TYPES = ("flat",)

_SIDECAR_SUFFIX = ".meta.json"


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    """Configuration and provenance of a saved :class:`FaissRetrievalIndex`.

    Attributes:
        dim: Embedding dimensionality.
        metric: Distance metric the index was built with.
        index_type: FAISS index flavor.
        embedding_model: Identifier of the model that produced the embeddings.
        embedding_version: Version of ``embedding_model``.
        preprocessing_version: Version of the preprocessing applied before
            embedding.
        created_at: UTC timestamp the index was constructed.
    """

    dim: int
    metric: Literal["l2", "ip", "cosine"]
    index_type: str
    embedding_model: str = ""
    embedding_version: str = ""
    preprocessing_version: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "dim": self.dim,
            "metric": self.metric,
            "index_type": self.index_type,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "preprocessing_version": self.preprocessing_version,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            dim=data["dim"],
            metric=data["metric"],
            index_type=data["index_type"],
            embedding_model=data.get("embedding_model", ""),
            embedding_version=data.get("embedding_version", ""),
            preprocessing_version=data.get("preprocessing_version", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def is_compatible_with(self, other: IndexMetadata) -> bool:
        """Return ``True`` if embeddings built under ``other`` may be mixed in."""
        return (
            self.dim == other.dim
            and self.metric == other.metric
            and self.index_type == other.index_type
            and self.embedding_model == other.embedding_model
            and self.embedding_version == other.embedding_version
            and self.preprocessing_version == other.preprocessing_version
        )


class FaissRetrievalIndex(RetrievalIndex):
    """A :class:`RetrievalIndex` backed by FAISS.

    Vector IDs are assigned sequentially by insertion order and are stable
    for the lifetime of the index (there is no removal at M19).

    Args:
        dim: Embedding dimensionality.
        metric: Distance metric. ``"cosine"`` is implemented as inner-product
            search over L2-normalized vectors.
        index_type: FAISS index flavor. Only ``"flat"`` is supported.
        embedding_model: Identifier of the model producing embeddings, stored
            in the index metadata for compatibility checks on load.
        embedding_version: Version of ``embedding_model``.
        preprocessing_version: Version of the preprocessing applied before
            embedding.
        faiss_module: Optional FAISS-like module, used by tests.

    Raises:
        RetrievalError: If ``index_type`` is not supported.
    """

    def __init__(
        self,
        dim: int,
        *,
        metric: Literal["l2", "ip", "cosine"] = "cosine",
        index_type: str = "flat",
        embedding_model: str = "",
        embedding_version: str = "",
        preprocessing_version: str = "",
        faiss_module: Any | None = None,
    ) -> None:
        """Initialize the index and build the underlying FAISS structure."""
        if index_type not in _SUPPORTED_INDEX_TYPES:
            msg = (
                f"Unsupported FAISS index_type {index_type!r}; "
                f"supported: {_SUPPORTED_INDEX_TYPES}"
            )
            raise RetrievalError(msg)
        self._dim = dim
        self._metric: Literal["l2", "ip", "cosine"] = metric
        self._index_type = index_type
        self._faiss = faiss_module or _import_faiss()
        self._metadata_meta = IndexMetadata(
            dim=dim,
            metric=metric,
            index_type=index_type,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            preprocessing_version=preprocessing_version,
        )
        self._index = self._build_index()
        self._metadata: list[dict[str, Any]] = []

    @property
    def metadata_info(self) -> IndexMetadata:
        """Return the index's configuration/provenance metadata."""
        return self._metadata_meta

    def add(self, embeddings: Embedding, metadata: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        """Add embeddings with associated metadata.

        Raises:
            RetrievalError: On dimension mismatch, non-finite values, or a
                length mismatch between ``embeddings`` and ``metadata``.
        """
        vectors = self._validate_embeddings(embeddings, expected_ndim=2)
        if vectors.shape[0] != len(metadata):
            msg = (
                f"embeddings has {vectors.shape[0]} rows but metadata has "
                f"{len(metadata)} entries"
            )
            raise RetrievalError(msg)
        if vectors.shape[0] == 0:
            return ()
        vectors = self._normalize_if_cosine(vectors)
        start_id = len(self._metadata)
        try:
            self._index.add(vectors)
        except Exception as exc:
            msg = f"Failed to add {vectors.shape[0]} embeddings to the FAISS index: {exc}"
            raise RetrievalError(msg) from exc
        self._metadata.extend(dict(m) for m in metadata)
        return tuple(range(start_id, start_id + vectors.shape[0]))

    def search(self, query: Embedding, top_k: int = 3) -> tuple[RetrievalMatch, ...]:
        """Return the ``top_k`` nearest historical matches to ``query``.

        Raises:
            RetrievalError: On dimension mismatch or search failure.
        """
        vector = self._validate_embeddings(query, expected_ndim=1).reshape(1, -1)
        vector = self._normalize_if_cosine(vector)
        try:
            distances, indices = self._index.search(vector, top_k)
        except Exception as exc:
            msg = f"FAISS search failed: {exc}"
            raise RetrievalError(msg) from exc
        matches: list[RetrievalMatch] = []
        for idx, dist in zip(indices[0], distances[0], strict=True):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[int(idx)]
            matches.append(
                RetrievalMatch(
                    vector_id=int(idx),
                    distance=float(dist),
                    dataset=str(meta.get("dataset", "")),
                    category=str(meta.get("category", "")),
                    defect_type=str(meta.get("defect_type", "")),
                    image_path=meta.get("image_path"),
                    metadata=dict(meta),
                )
            )
        return tuple(matches)

    def save(self, path: Path) -> None:
        """Persist the index and its metadata sidecar to ``path``.

        Raises:
            RetrievalError: On storage failure.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self._index, str(path))
            sidecar = {
                "index_metadata": self._metadata_meta.to_dict(),
                "records": self._metadata,
            }
            _sidecar_path(path).write_text(json.dumps(sidecar), encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to save FAISS index to {path}: {exc}"
            raise RetrievalError(msg) from exc

    def load(self, path: Path) -> None:
        """Load a previously saved index and its metadata sidecar from ``path``.

        Raises:
            RetrievalError: If the index is missing, corrupt, or was built
                with an incompatible embedding configuration.
        """
        path = Path(path)
        try:
            sidecar = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
            saved_meta = IndexMetadata.from_dict(sidecar["index_metadata"])
            if not self._metadata_meta.is_compatible_with(saved_meta):
                msg = (
                    f"Refusing to load index built with incompatible "
                    f"configuration: {saved_meta} != {self._metadata_meta}"
                )
                raise RetrievalError(msg)
            self._index = self._faiss.read_index(str(path))
            self._metadata = list(sidecar["records"])
        except RetrievalError:
            raise
        except Exception as exc:
            msg = f"Failed to load FAISS index from {path}: {exc}"
            raise RetrievalError(msg) from exc

    def _build_index(self) -> Any:
        """Construct the underlying FAISS index for the configured metric."""
        if self._metric == "l2":
            return self._faiss.IndexFlatL2(self._dim)
        return self._faiss.IndexFlatIP(self._dim)

    def _validate_embeddings(self, embeddings: Embedding, *, expected_ndim: int) -> Embedding:
        """Validate shape, dtype, and finiteness of ``embeddings``."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != expected_ndim:
            msg = f"Expected a {expected_ndim}D array, got shape {vectors.shape}"
            raise RetrievalError(msg)
        last_dim = vectors.shape[-1] if vectors.ndim > 0 else 0
        if vectors.size and last_dim != self._dim:
            msg = f"Expected embedding dimension {self._dim}, got {last_dim}"
            raise RetrievalError(msg)
        if vectors.size and not np.all(np.isfinite(vectors)):
            msg = "Embeddings contain NaN or infinite values"
            raise RetrievalError(msg)
        return vectors

    def _normalize_if_cosine(self, vectors: Embedding) -> Embedding:
        """L2-normalize rows when the index metric is ``"cosine"``."""
        if self._metric != "cosine" or vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return cast("Embedding", (vectors / norms).astype(np.float32))


def _sidecar_path(index_path: Path) -> Path:
    """Return the metadata sidecar path for a FAISS index file."""
    return index_path.with_name(index_path.name + _SIDECAR_SUFFIX)


def _import_faiss() -> Any:
    """Import FAISS lazily with a domain-specific error."""
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        msg = "faiss is not installed in the active environment"
        raise RetrievalError(msg) from exc


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    dim = 8
    index = FaissRetrievalIndex(dim, metric="cosine", embedding_model="smoke-test")

    base = rng.standard_normal((5, dim)).astype(np.float32)
    metadata = [
        {"dataset": "mvtec_ad", "category": "bottle", "defect_type": "crack"},
        {"dataset": "mvtec_ad", "category": "bottle", "defect_type": "scratch"},
        {"dataset": "visa", "category": "capsules", "defect_type": "poke"},
        {"dataset": "kolektorsdd2", "category": "surface", "defect_type": "blob"},
        {"dataset": "severstal", "category": "steel", "defect_type": "pit"},
    ]
    ids = index.add(base, metadata)
    assert ids == (0, 1, 2, 3, 4), ids

    query = base[2]
    results = index.search(query, top_k=1)
    assert results, "expected at least one match"
    assert results[0].vector_id == 2, f"nearest match should be itself, got {results[0]}"
    print(f"smoke test insert/query OK: nearest to row 2 is vector_id={results[0].vector_id}")

    tmp_path = Path("/tmp") / "faiss_index_smoke_test.faiss"
    index.save(tmp_path)

    reloaded = FaissRetrievalIndex(dim, metric="cosine", embedding_model="smoke-test")
    reloaded.load(tmp_path)
    reloaded_results = reloaded.search(query, top_k=1)
    assert reloaded_results[0].vector_id == 2, reloaded_results
    print("smoke test save/load OK")

    tmp_path.unlink(missing_ok=True)
    _sidecar_path(tmp_path).unlink(missing_ok=True)
