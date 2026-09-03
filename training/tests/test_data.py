"""Generic (never-named) dataset discovery -- discover_configs' fallback path."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from training.data import discover_configs, load_split


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.random.rand(32, 32, 3) * 255).astype("uint8"))


def test_generic_multi_category_dataset_is_discovered(tmp_path: Path) -> None:
    root = tmp_path / "my_widget"
    _write_image(root / "gadget" / "train" / "good" / "000.png")
    _write_image(root / "gadget" / "test" / "good" / "000.png")
    _write_image(root / "gadget" / "test" / "scratch" / "000.png")

    configs = discover_configs(tmp_path)
    keys = [c.key for c in configs]
    assert "my_widget/gadget" in keys


def test_generic_single_category_dataset_is_discovered(tmp_path: Path) -> None:
    root = tmp_path / "steel_panels"
    _write_image(root / "train" / "good" / "000.png")
    _write_image(root / "test" / "good" / "000.png")

    configs = discover_configs(tmp_path)
    assert [c.key for c in configs] == ["steel_panels"]


def test_generic_dataset_loads_with_correct_labels(tmp_path: Path) -> None:
    root = tmp_path / "my_widget"
    _write_image(root / "gadget" / "train" / "good" / "000.png")
    _write_image(root / "gadget" / "train" / "good" / "001.png")
    _write_image(root / "gadget" / "test" / "good" / "000.png")
    _write_image(root / "gadget" / "test" / "crack" / "000.png")

    config = next(c for c in discover_configs(tmp_path) if c.key == "my_widget/gadget")
    train, test = load_split(config, tmp_path)

    assert len(train) == 2
    assert sorted(test) == sorted(
        [
            (root / "gadget" / "test" / "good" / "000.png", False),
            (root / "gadget" / "test" / "crack" / "000.png", True),
        ]
    )


def test_generic_dataset_gets_default_geometry(tmp_path: Path) -> None:
    root = tmp_path / "unknown_corpus"
    _write_image(root / "train" / "good" / "000.png")

    config = next(c for c in discover_configs(tmp_path) if c.key == "unknown_corpus")
    assert (config.height, config.width) == (256, 256)
    assert config.position_aligned is True


def test_named_dataset_directory_is_not_rediscovered_generically(tmp_path: Path) -> None:
    # A directory literally named "mvtec" satisfies the MVTec-shaped test
    # already, so the dedicated mvtec branch finds it (expected - it is a
    # real MVTec-shaped corpus). What must not happen is *also* picking it
    # up a second time through the generic scan, producing a duplicate.
    root = tmp_path / "mvtec"
    _write_image(root / "some_category" / "train" / "good" / "000.png")

    configs = discover_configs(tmp_path)
    keys = [c.key for c in configs]
    assert keys == ["mvtec/some_category"]  # found once, not twice
    assert configs[0].dataset == "mvtec"


def test_hidden_directories_are_not_discovered(tmp_path: Path) -> None:
    _write_image(tmp_path / ".cache" / "train" / "good" / "000.png")
    assert discover_configs(tmp_path) == []
