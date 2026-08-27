"""Persisted per-run predictions.

The sweep is the expensive part, so it must only ever run once. Every run
writes its raw scores (and, where produced, its anomaly maps) to a compressed
archive keyed by ``regime/method/config``. Everything downstream reads those
instead of re-fitting:

* the dashboard draws ROC curves and score histograms from them;
* ``ensemble.py`` fuses methods that were never run together;
* any metric invented later can be recomputed without touching a GPU.

Maps are stored at a reduced resolution and as float16. Full-resolution float32
maps for 29 configurations across a zoo this size would run to hundreds of
gigabytes, and nothing downstream needs that precision -- the pixel metrics
resize to a common grid anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Anomaly maps are stored on a grid no larger than this on the long side.
MAP_STORE_SIZE = 256


@dataclass(frozen=True)
class RunArtifact:
    """Raw predictions for one ``(regime, method, config)`` run.

    Attributes:
        scores: ``(N,)`` image-level anomaly scores.
        labels: ``(N,)`` ground-truth anomaly labels.
        paths: ``(N,)`` source image paths, as strings, for the gallery.
        maps: ``(N, h, w)`` anomaly maps, or ``None`` if the method produced none.
    """

    scores: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    maps: np.ndarray | None

    def __len__(self) -> int:
        """Number of scored images."""
        return int(self.scores.shape[0])


def artifact_path(root: Path, regime: str, method: str, config_key: str, seed: int) -> Path:
    """Location of one run's archive.

    Keyed by seed too: with the 3-seed repeats, three separate fits produce
    three separate score/label/map sets for the same (regime, method,
    config) -- without the seed in the filename, the second and third fit
    would each silently overwrite the previous seed's saved archive.
    """
    slug = config_key.replace("/", "_")
    return root / regime / f"{method}__{slug}__seed{seed}.npz"


def save_artifact(
    path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    maps: np.ndarray | None = None,
) -> None:
    """Write one run's predictions.

    Args:
        path: Destination ``.npz``.
        scores: Image-level anomaly scores.
        labels: Ground-truth anomaly labels.
        paths: Source image paths, in the same order.
        maps: Optional anomaly maps; downsampled and cast to float16 here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=bool),
        "paths": np.asarray(paths, dtype=object),
    }
    if maps is not None:
        payload["maps"] = _shrink_maps(np.asarray(maps)).astype(np.float16)
    np.savez_compressed(path, **payload)


def load_artifact(path: Path) -> RunArtifact | None:
    """Read one run's predictions, or ``None`` if the archive is absent."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return RunArtifact(
            scores=data["scores"],
            labels=data["labels"],
            paths=data["paths"],
            maps=data["maps"].astype(np.float32) if "maps" in data else None,
        )


def _shrink_maps(maps: np.ndarray) -> np.ndarray:
    """Downsample maps so the long side is at most :data:`MAP_STORE_SIZE`.

    Uses OpenCV area interpolation, which preserves peak location better than
    naive striding -- and peak location is the whole point of a heatmap.
    """
    if maps.ndim != 3:
        return maps
    height, width = int(maps.shape[1]), int(maps.shape[2])
    longest = max(height, width)
    if longest <= MAP_STORE_SIZE:
        return maps

    import cv2

    scale = MAP_STORE_SIZE / longest
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.stack(
        [cv2.resize(m.astype(np.float32), target, interpolation=cv2.INTER_AREA) for m in maps]
    )
