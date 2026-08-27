"""Batched image loading for the sweep.

``image_io.load_rgb`` reads one image at a time, which is fine for a single
fit but becomes the bottleneck when the same corpus is read once per method in
the zoo. This wraps it in a ``Dataset``/``DataLoader`` so decode and resize run
on worker processes while the GPU is busy on the previous batch.
"""

from __future__ import annotations

from pathlib import Path

import torch
from image_io import load_rgb
from torch.utils.data import DataLoader, Dataset


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
