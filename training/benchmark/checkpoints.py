"""Persisted fitted models.

Every job's scorer -- whether it is a PatchCore memory bank, a PaDiM Gaussian,
a DFM PCA basis, or a trained Dinomaly decoder -- is a plain ``torch.nn.Module``
with no unpicklable state (no open handles, no stored hooks, no live CUDA
streams). That means a whole-object save round-trips cleanly for every method
in the zoo without any per-family bespoke serialization: the same two
functions here work for all of them.

Saving happens once per fit, right after ``spec.fit()`` returns and before the
scorer is used for scoring or goes out of scope -- see ``regimes.py``. This is
what lets the deployment-cost pass (``cost.py``) measure real inference
latency/VRAM/params by loading a checkpoint and timing its forward pass,
instead of re-fitting just to get a model to time.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def checkpoint_path(root: Path, regime: str, method: str, config_key: str) -> Path:
    """Location of one run's saved model.

    Mirrors ``artifacts.artifact_path``'s ``{regime}/{method}__{slug}`` layout
    so the two persisted-per-run archives (predictions, model) stay easy to
    correlate on disk.
    """
    slug = config_key.replace("/", "_")
    return root / regime / f"{method}__{slug}.pt"


def save_checkpoint(scorer: nn.Module, path: Path) -> None:
    """Write a fitted scorer to disk.

    Args:
        scorer: The fitted model, straight out of ``spec.fit()``.
        path: Destination ``.pt``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scorer, path)


def load_checkpoint(path: Path, device: str = "cuda") -> nn.Module | None:
    """Read a saved scorer back, or ``None`` if the checkpoint is absent.

    Args:
        path: Checkpoint written by :func:`save_checkpoint`.
        device: Device to map the restored tensors onto.

    Returns:
        The restored scorer in eval mode, ready to score or time, or ``None``
        if nothing was saved at ``path`` (e.g. the run failed before fitting).
    """
    if not path.exists():
        return None
    scorer = torch.load(path, map_location=device, weights_only=False)
    scorer.eval()
    return scorer
