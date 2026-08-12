"""Bounded local image archival (Milestone M4).

This module provides a simple, bounded image store for the current M4 scope. It
archives raw image bytes to a local directory and returns a stable reference
that is recorded in the inspection result's ``image_refs``.

The store is deliberately simple at M4: it writes each image to a file named by
its frame id and enforces a maximum number of retained images by evicting the
oldest files. Advanced V1 buffering / WAL behavior is explicitly out of scope
and belongs to Milestone M17.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from adaptivevision.common.errors import AdaptiveVisionError


class ImageStoreError(AdaptiveVisionError):
    """Failure to archive or retrieve an image."""


class LocalImageStore:
    """A bounded, local, on-disk image archive.

    Args:
        directory: Directory to store image files in. Created if missing.
        max_images: Maximum number of images to retain. When exceeded, the
            oldest files are evicted.
    """

    def __init__(self, directory: str | Path, *, max_images: int = 1000) -> None:
        """Initialize the store."""
        self._directory = Path(directory)
        self._max_images = max_images
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """Return the archive directory."""
        return self._directory

    def archive(self, frame_id: str, data: bytes) -> str:
        """Archive image ``data`` under ``frame_id``.

        Args:
            frame_id: Unique identifier of the frame (used as the file name).
            data: Raw image bytes to persist.

        Returns:
            A stable reference to the archived image.

        Raises:
            ImageStoreError: If the image cannot be written.
        """
        if not frame_id:
            msg = "frame_id must not be empty"
            raise ImageStoreError(msg)
        path = self._directory / f"{frame_id}.bin"
        try:
            path.write_bytes(data)
        except OSError as exc:
            msg = f"Failed to archive image {frame_id!r}: {exc}"
            raise ImageStoreError(msg) from exc
        self._evict_oldest()
        return str(path)

    def resolve(self, reference: str) -> Path:
        """Resolve an image reference to its on-disk path.

        Args:
            reference: The reference returned by :meth:`archive`.

        Returns:
            The :class:`~pathlib.Path` of the archived image.

        Raises:
            ImageStoreError: If the referenced image is missing.
        """
        path = Path(reference)
        if not path.exists():
            msg = f"Archived image not found: {reference!r}"
            raise ImageStoreError(msg)
        return path

    def _evict_oldest(self) -> None:
        """Evict the oldest files when the store exceeds its bound."""
        files = sorted(self._directory.glob("*.bin"), key=lambda p: p.stat().st_mtime)
        while len(files) > self._max_images:
            oldest = files.pop(0)
            with contextlib.suppress(OSError):
                oldest.unlink()
