"""Dataset loaders for the anomaly-detection training pipeline.

Each loader returns a pair: the list of *normal-only* training image paths,
and a labeled test split as ``(path, is_anomalous)`` tuples. None of this
touches the production ``src/adaptivevision`` tree -- it only prepares paths
and labels consumed by ``train_anomaly_model.py``.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2


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
    name, deliberately not surfaced here (see ``data.loco_defect_kind``,
    used by regimes.py's structural/logical breakdown, not by this loader).

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
