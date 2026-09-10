"""Data layer: dataset discovery, path/label loading, masks, and batched image I/O.

Everything that turns a directory on disk into images and labels a model can
train or score against, for MVTec AD, MVTec LOCO, VisA, KolektorSDD2, and
Severstal, plus any other dataset that happens to follow the same
``<category>/train/good/`` layout MVTec popularized.

None of this touches the production ``src/adaptivevision`` tree -- it only
prepares paths, labels, and masks consumed by ``training.legacy``,
``training.sweep``, and the rest of the training/benchmark pipeline.

Organized bottom-up:

1. Per-corpus path/label loaders (``mvtec_paths``, ``visa_paths``, ...).
2. :class:`DatasetConfig` and :func:`discover_configs`, which wrap those
   loaders into one uniform, addressable interface (``mvtec/bottle``).
3. Ground-truth mask loading, normalized across each corpus's own convention.
4. Single-image and batched (``Dataset``/``DataLoader``) image I/O.
5. Split subsampling -- capping a fit/test split to a budget, seeded and
   class-balanced. Lives here rather than in ``training.sweep`` so both it
   and ``training.evaluate`` (whose metrics-backfill pass has to reproduce
   the exact same pixel-metrics sample the sweep used) can depend on it
   without the two importing each other.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Per-corpus path/label loaders
# -----------------------------------------------------------------------------


def mvtec_paths(category_dir: Path) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Load MVTec AD paths for one category directory (e.g. ``mvtec/bottle``)."""
    train = sorted((category_dir / "train" / "good").glob("*.png"))
    test: list[tuple[Path, bool]] = []
    test_dir = category_dir / "test"
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        is_anomalous = class_dir.name != "good"
        for image_path in sorted(class_dir.glob("*.png")):
            test.append((image_path, is_anomalous))
    return train, test


def mvtec_loco_paths(category_dir: Path) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Load MVTec LOCO paths for one category directory.

    Same train/test convention as vanilla MVTec AD (:func:`mvtec_paths`) --
    ``train/good`` + one ``test/<subfolder>`` per class, ``good`` normal and
    everything else anomalous -- so this just reuses that loader. LOCO's own
    structural-vs-logical distinction lives only in the ``test/<subfolder>``
    name, deliberately not surfaced here (see :func:`loco_defect_kind`, used
    by the sweep's structural/logical regime breakdown, not by this loader).

    UNVERIFIED: assumes the same train/test/ground_truth directory shape as
    vanilla MVTec AD holds for LOCO too -- not yet checked against a real
    download. If it differs, only this function needs to change.
    """
    return mvtec_paths(category_dir)


def visa_paths(root: Path, object_name: str) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Load VisA paths for one object using ``split_csv/1cls.csv``."""
    csv_path = root / "split_csv" / "1cls.csv"
    train: list[Path] = []
    test: list[tuple[Path, bool]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["object"] != object_name:
                continue
            image_path = root / row["image"]
            if row["split"] == "train":
                train.append(image_path)
            else:
                test.append((image_path, row["label"] == "anomaly"))
    return train, test


def kolektor_paths(root: Path) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Load KolektorSDD2 paths, labeling by whether the GT mask is non-zero."""
    train = _kolektor_split(root / "train")
    test_all = _kolektor_split(root / "test")
    train_normal = [path for path, is_anomalous in train if not is_anomalous]
    return train_normal, test_all


def _kolektor_split(split_dir: Path) -> list[tuple[Path, bool]]:
    labeled: list[tuple[Path, bool]] = []
    for image_path in sorted(split_dir.glob("*.png")):
        if image_path.stem.endswith("_GT"):
            continue
        mask_path = split_dir / f"{image_path.stem}_GT.png"
        is_anomalous = False
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            is_anomalous = mask is not None and bool(mask.max() > 0)
        labeled.append((image_path, is_anomalous))
    return labeled


def severstal_paths(root: Path) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Load Severstal paths, labeling by presence of a defect row in train.csv."""
    images_dir = root / "train_images"
    defective: set[str] = set()
    with (root / "train.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["EncodedPixels"].strip():
                defective.add(row["ImageId"])

    train: list[Path] = []
    test: list[tuple[Path, bool]] = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        is_anomalous = image_path.name in defective
        if is_anomalous:
            test.append((image_path, True))
        else:
            train.append(image_path)

    # Severstal has no separate labeled-normal test split; hold out 10% of the
    # normal images as normal test examples so evaluation has both classes.
    holdout = max(1, len(train) // 10)
    test.extend((path, False) for path in train[-holdout:])
    train = train[:-holdout]
    return train, test


def subsample_to_prevalence(
    test_split: list[tuple[Path, bool]], target_prevalence: float, seed: int
) -> list[tuple[Path, bool]]:
    """Downsample the majority class so the anomalous rate hits ``target_prevalence``.

    Severstal's test split runs ~92% positive (see :func:`severstal_paths`) --
    far more prevalence-sensitive metrics like AP/F1 than the other corpora,
    which sit much lower. This downsamples the anomalous images (the
    majority class here) to hit the target rate while keeping every normal
    image; it never upsamples, since that would fabricate data.

    Args:
        test_split: The full test split.
        target_prevalence: Desired fraction of anomalous images, in ``(0, 1)``.
        seed: Seed for which anomalous images get kept.

    Returns:
        A new, shuffled test split at (approximately) the target prevalence.
        Returns the input unchanged if either class is empty.
    """
    normal = [item for item in test_split if not item[1]]
    anomalous = [item for item in test_split if item[1]]
    if not normal or not anomalous:
        return list(test_split)

    # Solve for the anomalous count that hits target_prevalence while every
    # normal image is kept: target = n_anom / (n_normal + n_anom).
    target_anomalous = round(target_prevalence * len(normal) / max(1e-9, 1 - target_prevalence))
    target_anomalous = max(1, min(target_anomalous, len(anomalous)))

    rng = random.Random(seed)
    shuffled_anomalous = list(anomalous)
    rng.shuffle(shuffled_anomalous)
    kept = normal + shuffled_anomalous[:target_anomalous]
    rng.shuffle(kept)
    return kept


DATASET_LOADERS = {
    "mvtec": mvtec_paths,
    "visa": visa_paths,
    "kolektor": kolektor_paths,
    "severstal": severstal_paths,
    "mvtec_loco": mvtec_loco_paths,
}

# -----------------------------------------------------------------------------
# Dataset configuration and discovery
# -----------------------------------------------------------------------------

#: Native aspect ratios differ a lot across these datasets (Severstal strips
#: are ~6:1, Kolektor panels ~2.8:1). Resizing everything to a square would
#: squash the very defects we are trying to detect, so each dataset keeps the
#: (height, width) tuned to its real proportions -- and every method in the
#: zoo sees the same geometry, which is what makes the comparison fair.
DEFAULT_SIZE: dict[str, tuple[int, int]] = {
    "mvtec": (256, 256),
    "visa": (256, 256),
    "kolektor": (256, 96),
    "severstal": (128, 800),
    "mvtec_loco": (256, 256),
}

#: Fallback geometry/prior for a dataset discovered generically (see
#: :func:`_discover_generic_configs`), whose true aspect ratio and alignment
#: this module has no way to know in advance. Square and camera-aligned
#: matches the common case (MVTec/VisA/LOCO-shaped corpora); a genuinely
#: unaligned or extreme-aspect-ratio dataset should get its own
#: ``DEFAULT_SIZE``/``POSITION_ALIGNED`` entry once identified, the same way
#: Severstal and Kolektor already do.
_GENERIC_DEFAULT_SIZE: tuple[int, int] = (256, 256)
_GENERIC_POSITION_ALIGNED: bool = True

#: The five corpora with dedicated loaders/roots - never re-discovered
#: generically even if they also happen to satisfy the generic folder shape.
_NAMED_DATASETS = frozenset({"mvtec", "visa", "kolektor", "severstal", "mvtec_loco"})

#: Per-patch-position modeling assumes camera-aligned parts. Severstal crops
#: are unaligned sections of continuous steel strip, so position-agnostic
#: (pooled) statistics are the correct prior there. Methods that care read
#: this off the config; methods that are inherently position-agnostic ignore it.
POSITION_ALIGNED: dict[str, bool] = {
    "mvtec": True,
    "visa": True,
    "kolektor": True,
    "severstal": False,
    "mvtec_loco": True,
}


@dataclass(frozen=True)
class DatasetConfig:
    """One addressable benchmark target.

    Attributes:
        dataset: Dataset family key (``mvtec``/``visa``/``kolektor``/``severstal``).
        category: Category or object name, or ``None`` for single-category datasets.
        height: Model input height used by every method for this config.
        width: Model input width used by every method for this config.
        position_aligned: Whether patch position is semantically stable here.
    """

    dataset: str
    category: str | None
    height: int
    width: int
    position_aligned: bool

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``mvtec/bottle`` or ``severstal``."""
        return f"{self.dataset}/{self.category}" if self.category else self.dataset

    @property
    def slug(self) -> str:
        """Filesystem-safe form of :attr:`key`."""
        return self.key.replace("/", "_")


def _root_for(dataset: str, data_root: Path) -> Path:
    """Map a dataset family to its directory under ``data_root``.

    A generically-discovered dataset (see :func:`_discover_generic_configs`)
    is not in this table by construction - its directory name *is* its
    dataset key, so it resolves by simple concatenation instead.
    """
    named = {
        "mvtec": data_root / "mvtec",
        "visa": data_root / "VisA_20220922",
        "kolektor": data_root / "KolektorSDD2",
        "severstal": data_root / "severstal-steel-defect-detection",
        # UNVERIFIED: the official archive's extracted top-level directory
        # name has not been confirmed against a real download. Adjust this
        # one line once MVTec LOCO is actually downloaded, if it differs.
        "mvtec_loco": data_root / "mvtec_loco",
    }
    return named.get(dataset, data_root / dataset)


def discover_configs(data_root: Path) -> list[DatasetConfig]:
    """Find every dataset configuration present under ``data_root``.

    Missing datasets are skipped rather than raising, so the sweep still runs
    on whatever subset of the four corpora is actually downloaded.

    Args:
        data_root: Directory holding ``mvtec/``, ``VisA_20220922/``,
            ``KolektorSDD2/`` and ``severstal-steel-defect-detection/``.

    Returns:
        Configs sorted by :attr:`DatasetConfig.key`.
    """
    configs: list[DatasetConfig] = []

    mvtec_root = _root_for("mvtec", data_root)
    if mvtec_root.is_dir():
        for category_dir in sorted(mvtec_root.iterdir()):
            if (category_dir / "train" / "good").is_dir():
                configs.append(_make("mvtec", category_dir.name))

    visa_root = _root_for("visa", data_root)
    if (visa_root / "split_csv" / "1cls.csv").is_file():
        for object_dir in sorted(visa_root.iterdir()):
            if (object_dir / "Data" / "Images").is_dir():
                configs.append(_make("visa", object_dir.name))

    if (_root_for("kolektor", data_root) / "train").is_dir():
        configs.append(_make("kolektor", None))

    if (_root_for("severstal", data_root) / "train.csv").is_file():
        configs.append(_make("severstal", None))

    loco_root = _root_for("mvtec_loco", data_root)
    if loco_root.is_dir():
        for category_dir in sorted(loco_root.iterdir()):
            if (category_dir / "train" / "good").is_dir():
                configs.append(_make("mvtec_loco", category_dir.name))

    configs.extend(_discover_generic_configs(data_root))

    return sorted(configs, key=lambda config: config.key)


def _discover_generic_configs(data_root: Path) -> list[DatasetConfig]:
    """Find datasets under ``data_root`` that follow the MVTec-style layout
    (``<category>/train/good/`` for several categories, or ``train/good/``
    directly for a single-category set) but are not one of the five named
    corpora above.

    This is what lets a genuinely new dataset "just work": drop it under
    ``data_root`` in that shape and it is picked up with no code change,
    loaded via the same :func:`mvtec_paths` reader MVTec AD uses.
    """
    if not data_root.is_dir():
        return []

    discovered: list[DatasetConfig] = []
    named_roots = {_root_for(name, data_root) for name in _NAMED_DATASETS}
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir() or entry in named_roots or entry.name.startswith("."):
            continue
        if (entry / "train" / "good").is_dir():
            # Single-category: <entry>/train/good/, like Kolektor's shape.
            discovered.append(_make(entry.name, None))
            continue
        for category_dir in sorted(entry.iterdir()):
            if category_dir.is_dir() and (category_dir / "train" / "good").is_dir():
                discovered.append(_make(entry.name, category_dir.name))
    return discovered


def _make(dataset: str, category: str | None) -> DatasetConfig:
    """Build a config with this dataset family's default geometry and prior.

    Falls back to :data:`_GENERIC_DEFAULT_SIZE`/:data:`_GENERIC_POSITION_ALIGNED`
    for a dataset outside the five named corpora - see
    :func:`_discover_generic_configs`.
    """
    height, width = DEFAULT_SIZE.get(dataset, _GENERIC_DEFAULT_SIZE)
    return DatasetConfig(
        dataset=dataset,
        category=category,
        height=height,
        width=width,
        position_aligned=POSITION_ALIGNED.get(dataset, _GENERIC_POSITION_ALIGNED),
    )


def parse_config(key: str, data_root: Path) -> DatasetConfig:
    """Resolve a ``dataset[/category]`` key against the configs on disk.

    Raises:
        SystemExit: If ``key`` matches no discovered configuration.
    """
    for config in discover_configs(data_root):
        if config.key == key:
            return config
    msg = f"Unknown dataset config: {key!r}"
    raise SystemExit(msg)


def load_split(
    config: DatasetConfig, data_root: Path
) -> tuple[list[Path], list[tuple[Path, bool]]]:
    """Return ``(normal_train_paths, labeled_test_split)`` for ``config``.

    Args:
        config: The configuration to load.
        data_root: Directory holding the dataset corpora.

    Returns:
        Normal-only training paths, and ``(path, is_anomalous)`` test pairs.
    """
    root = _root_for(config.dataset, data_root)
    if config.dataset not in DATASET_LOADERS:
        # Generically-discovered dataset (see _discover_generic_configs):
        # always the MVTec-shaped reader, since that is the shape detection
        # itself required.
        return mvtec_paths(root / config.category if config.category else root)
    loader = DATASET_LOADERS[config.dataset]
    if config.dataset in ("mvtec", "mvtec_loco"):
        return loader(root / str(config.category))
    if config.dataset == "visa":
        return loader(root, str(config.category))
    return loader(root)


def loco_defect_kind(path: Path) -> str:
    """``"good"`` / ``"structural"`` / ``"logical"``, derived from a LOCO test path.

    Derived lazily from the path (mirroring how :func:`_mvtec_mask` already
    gets a MVTec AD image's defect-type folder from ``path.parent.name``)
    rather than widening the shared ``(Path, bool)`` test-split tuple every
    other loader and regime function uses -- only LOCO's structural/logical
    breakdown (the sweep's ``evaluate_loco_breakdown``) needs this, so it
    stays a leaf lookup instead of a change that ripples through the whole
    loading/scoring pipeline.

    UNVERIFIED: assumes LOCO's ``test/`` layout is ``good/``,
    ``structural_anomalies/``, ``logical_anomalies/`` (the official archive's
    documented convention) -- not yet checked against a real download.
    """
    name = path.parent.name
    if name == "good":
        return "good"
    return "logical" if "logical" in name else "structural"


# -----------------------------------------------------------------------------
# Ground-truth defect masks, normalized across four different conventions
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Single-image and batched image I/O
# -----------------------------------------------------------------------------


def bgr_to_model_input(bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a BGR array (as OpenCV decodes it) to the model's ``(3, H, W)`` contract.

    Returns:
        A ``(3, height, width)`` float32 array with values in ``[0, 255]``,
        matching the channel-first, no-batch-dim convention consumed by
        ``ThresholdAnomalyDetector``.
    """
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32)


def load_rgb(path: Path, height: int, width: int) -> np.ndarray:
    """Read ``path`` as BGR and convert to the model's ``(3, height, width)`` contract."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {path}"
        raise FileNotFoundError(msg)
    return bgr_to_model_input(image, height, width)


class ImagePathDataset(Dataset):
    """Decodes images on demand into the ``(3, H, W)`` float32 model contract.

    Args:
        paths: Image files to read.
        height: Target height.
        width: Target width.
    """

    def __init__(self, paths: list[Path], height: int, width: int) -> None:
        """Store the path list and target geometry."""
        self.paths = list(paths)
        self.height = height
        self.width = width

    def __len__(self) -> int:
        """Number of images."""
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return image ``index`` as a ``(3, H, W)`` float32 tensor in ``[0, 255]``."""
        return torch.from_numpy(load_rgb(self.paths[index], self.height, self.width))


def image_loader(
    paths: list[Path],
    height: int,
    width: int,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Build a deterministic, non-shuffling loader over ``paths``.

    Order is preserved so returned scores line up with the caller's labels.

    Args:
        paths: Image files to read.
        height: Target height.
        width: Target width.
        batch_size: Images per batch.
        num_workers: Worker processes; ``0`` loads in the main process.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    return DataLoader(
        ImagePathDataset(paths, height, width),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


# -----------------------------------------------------------------------------
# Split subsampling
# -----------------------------------------------------------------------------


def subsample_fit(paths: list[Path], limit: int, seed: int) -> list[Path]:
    """Take a seeded random subset of the normal training images.

    Random rather than head-of-list: several corpora are ordered by capture
    session, so the first N images can share a lighting or batch artifact that
    the rest of the set does not.
    """
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def subsample_test(
    split: list[tuple[Path, bool]], limit: int, seed: int
) -> list[tuple[Path, bool]]:
    """Take a class-balanced seeded subset of the test split.

    Capping without balancing would silently change the positive rate, which
    moves average precision and F1 and makes configs incomparable.
    """
    if limit <= 0 or len(split) <= limit:
        return list(split)
    rng = random.Random(seed)
    normal = [item for item in split if not item[1]]
    anomalous = [item for item in split if item[1]]
    rng.shuffle(normal)
    rng.shuffle(anomalous)
    per_class = max(1, limit // 2)
    kept = normal[:per_class] + anomalous[:per_class]
    rng.shuffle(kept)
    return kept


#: Cap on images whose masks are loaded for pixel metrics. Decoding and
#: resizing masks for all ~7,250 Severstal test images, once per method, would
#: dominate the sweep; a class-balanced sample estimates the same quantity.
_MAX_PIXEL_IMAGES = 400


def pixel_sample_indices(test_split: list[tuple[Path, bool]], seed: int) -> list[int]:
    """Indices of a class-balanced sample capped at :data:`_MAX_PIXEL_IMAGES`.

    Normal images are included deliberately: a model that lights up clean
    parts should be punished on pixel AUROC, and a defect-only sample would
    hide exactly that.

    Shared by ``training.sweep`` (the original pixel-metrics pass) and
    ``training.evaluate`` (the metrics-backfill pass, which needs the exact
    same sample to produce a comparable backfilled AUPIMO number).
    """
    if len(test_split) <= _MAX_PIXEL_IMAGES:
        return list(range(len(test_split)))
    rng = random.Random(seed)
    anomalous = [i for i, (_, is_anom) in enumerate(test_split) if is_anom]
    normal = [i for i, (_, is_anom) in enumerate(test_split) if not is_anom]
    rng.shuffle(anomalous)
    rng.shuffle(normal)
    per_class = _MAX_PIXEL_IMAGES // 2
    return sorted(anomalous[:per_class] + normal[:per_class])
