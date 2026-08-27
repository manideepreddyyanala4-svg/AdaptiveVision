"""Enumeration of every dataset configuration available on disk.

``datasets.py`` knows how to turn one dataset directory into
``(normal_train_paths, labeled_test_split)``. This module wraps those loaders
into a uniform :class:`DatasetConfig` and discovers *all* configurations --
15 MVTec AD categories, 12 VisA objects, KolektorSDD2 and Severstal -- so a
sweep can address them by a single stable key such as ``mvtec/bottle``.

Beyond those five named corpora, :func:`discover_configs` also picks up any
*other* directory under ``data_root`` that happens to follow the same
``<category>/train/good/`` (or ``train/good/`` directly, for a
single-category set) layout MVTec popularized -- the most common convention
industrial anomaly-detection datasets use. Such a directory needs zero code
changes here: it is loaded with the same :func:`datasets.mvtec_paths` reader
MVTec AD itself uses. A dataset in a genuinely different shape (VisA's CSV
split, Severstal's mask CSV, ...) still needs its own small loader function
in ``datasets.py`` and an entry in ``DATASET_LOADERS`` - no amount of
generic detection can guess an unfamiliar label format, only recognize a
familiar folder shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import DATASET_LOADERS, mvtec_paths

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
    loaded via the same :func:`datasets.mvtec_paths` reader MVTec AD uses.
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

    Derived lazily from the path (mirroring how ``masks.py``'s ``_mvtec_mask``
    already gets a MVTec AD image's defect-type folder from ``path.parent.name``)
    rather than widening the shared ``(Path, bool)`` test-split tuple every
    other loader and regime function uses -- only LOCO's structural/logical
    breakdown (regimes.py's ``evaluate_loco_breakdown``) needs this, so it
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
