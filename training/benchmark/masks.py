"""Ground-truth defect masks, normalized across four different conventions.

Every corpus stores localization ground truth differently: MVTec puts it in a
parallel ``ground_truth/`` tree with a ``_mask`` suffix, VisA names it in the
split CSV, KolektorSDD2 sits it next to the image with a ``_GT`` suffix, and
Severstal encodes it as run-length strings in a CSV. This module hides all of
that behind one call, so pixel metrics and the heatmap gallery do not each
have to know four layouts.

A normal image has no mask file anywhere in these corpora; that is not missing
data, it is an all-zero mask, and it is returned as one.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from benchmark.data import DatasetConfig

#: Severstal images are a fixed size and its RLE is column-major over them.
_SEVERSTAL_SHAPE = (256, 1600)


def load_mask(
    path: Path, config: DatasetConfig, data_root: Path, height: int, width: int
) -> np.ndarray | None:
    """Load the defect mask for one image, resized to ``(height, width)``.

    Args:
        path: The image whose mask is wanted.
        config: Configuration the image belongs to.
        data_root: Directory holding the dataset corpora.
        height: Target mask height.
        width: Target mask width.

    Returns:
        A boolean ``(height, width)`` array, or ``None`` if this corpus
        provides no localization ground truth for the image at all.
    """
    raw = _load_raw_mask(path, config, data_root)
    if raw is None:
        return None
    if raw.shape != (height, width):
        raw = cv2.resize(raw.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return raw.astype(bool)


def _load_raw_mask(path: Path, config: DatasetConfig, data_root: Path) -> np.ndarray | None:
    """Dispatch to the per-corpus mask convention."""
    if config.dataset == "mvtec":
        return _mvtec_mask(path)
    if config.dataset == "visa":
        return _visa_mask(path, data_root)
    if config.dataset == "kolektor":
        return _kolektor_mask(path)
    if config.dataset == "severstal":
        return _severstal_mask(path, data_root)
    if config.dataset == "mvtec_loco":
        return _mvtec_loco_mask(path)
    return None


def _read_binary(mask_path: Path) -> np.ndarray | None:
    """Read a mask image as a boolean array, or ``None`` if unreadable."""
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 0


def _mvtec_mask(path: Path) -> np.ndarray | None:
    """Resolve ``.../test/<defect>/000.png`` to ``.../ground_truth/<defect>/000_mask.png``."""
    defect_class = path.parent.name
    category_dir = path.parent.parent.parent
    if defect_class == "good":
        return np.zeros(_probe_shape(path), dtype=bool)
    mask_path = category_dir / "ground_truth" / defect_class / f"{path.stem}_mask.png"
    return _read_binary(mask_path) if mask_path.exists() else None


def _mvtec_loco_mask(path: Path) -> np.ndarray | None:
    """Resolve one LOCO test image to its ground-truth mask(s), OR-combined.

    Unlike vanilla MVTec AD's one mask per image, LOCO ships one mask file
    per ground-truth *component* under ``ground_truth/<defect>/<stem>/`` --
    a directory of masks, since a logical anomaly can involve several
    simultaneous irregularities. OR-combining every mask found there into
    one boolean array matches how every other loader here (and the pixel
    metrics that consume it) expect a single mask per image; revisit this
    combination choice once real LOCO masks have been inspected.

    UNVERIFIED: assumes this ``ground_truth/<defect>/<stem>/*.png``
    directory-of-masks convention -- not yet checked against a real download.
    """
    defect_class = path.parent.name
    category_dir = path.parent.parent.parent
    if defect_class == "good":
        return np.zeros(_probe_shape(path), dtype=bool)

    mask_dir = category_dir / "ground_truth" / defect_class / path.stem
    if not mask_dir.is_dir():
        return None

    combined: np.ndarray | None = None
    for mask_path in sorted(mask_dir.glob("*.png")):
        component = _read_binary(mask_path)
        if component is None:
            continue
        combined = component if combined is None else (combined | component)
    return combined


def _visa_mask(path: Path, data_root: Path) -> np.ndarray | None:
    """Look the mask up in ``split_csv/1cls.csv``, which names it per row."""
    root = data_root / "VisA_20220922"
    lookup = _visa_mask_index(root)
    relative = _relative_to(path, root)
    if relative is None:
        return None
    mask_relative = lookup.get(relative)
    if mask_relative is None:
        # Present in the split with an empty mask cell: a normal image.
        return np.zeros(_probe_shape(path), dtype=bool) if relative in lookup else None
    mask_path = root / mask_relative
    return _read_binary(mask_path) if mask_path.exists() else None


@lru_cache(maxsize=4)
def _visa_mask_index(root: Path) -> dict[str, str | None]:
    """Map each VisA image path to its mask path, ``None`` where the cell is empty."""
    index: dict[str, str | None] = {}
    csv_path = root / "split_csv" / "1cls.csv"
    if not csv_path.exists():
        return index
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mask = (row.get("mask") or "").strip()
            index[row["image"].replace("\\", "/")] = mask or None
    return index


def _kolektor_mask(path: Path) -> np.ndarray | None:
    """Read the sibling ``<stem>_GT.png``."""
    mask_path = path.parent / f"{path.stem}_GT.png"
    return _read_binary(mask_path) if mask_path.exists() else None


def _severstal_mask(path: Path, data_root: Path) -> np.ndarray | None:
    """Decode the run-length encoding for this image, unioned over defect classes."""
    encodings = _severstal_rle_index(data_root / "severstal-steel-defect-detection")
    runs = encodings.get(path.name)
    mask = np.zeros(_SEVERSTAL_SHAPE, dtype=bool)
    if not runs:
        return mask
    flat = mask.reshape(-1, order="F")
    for encoded in runs:
        values = np.array(encoded.split(), dtype=np.int64)
        starts, lengths = values[0::2] - 1, values[1::2]
        for start, length in zip(starts, lengths, strict=True):
            flat[start : start + length] = True
    return flat.reshape(_SEVERSTAL_SHAPE, order="F")


@lru_cache(maxsize=4)
def _severstal_rle_index(root: Path) -> dict[str, list[str]]:
    """Group every RLE string in ``train.csv`` by image name."""
    index: dict[str, list[str]] = {}
    csv_path = root / "train.csv"
    if not csv_path.exists():
        return index
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            encoded = row["EncodedPixels"].strip()
            if encoded:
                index.setdefault(row["ImageId"], []).append(encoded)
    return index


def _relative_to(path: Path, root: Path) -> str | None:
    """POSIX-style path relative to ``root``, or ``None`` if outside it."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _probe_shape(path: Path) -> tuple[int, int]:
    """Native ``(height, width)`` of an image, for synthesizing an empty mask."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (1, 1) if image is None else (int(image.shape[0]), int(image.shape[1]))


def has_masks(config: DatasetConfig) -> bool:
    """Whether this corpus provides localization ground truth at all."""
    return config.dataset in {"mvtec", "visa", "kolektor", "severstal", "mvtec_loco"}
