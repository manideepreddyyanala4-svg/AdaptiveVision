"""The method registry: one uniform interface over the whole zoo.

Every entry -- whether it is a training-free memory bank implemented here or a
gradient-trained model delegated to Anomalib -- exposes the same two steps:
fit on normal-only images, then score a labeled test split. The runner never
needs to know which family it is driving, which is what keeps the leaderboard
an apples-to-apples comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from benchmark.data import DatasetConfig


@dataclass
class RunOptions:
    """Knobs shared by every method in a sweep.

    Attributes:
        device: Torch device string.
        max_fit_images: Cap on normal training images used to fit. ``0`` means
            use every available image.
        max_test_images: Cap on test images scored, applied class-balanced.
            ``0`` means score the whole split.
        batch_size: Images per forward pass; reduced automatically for
            VRAM-heavy backbones.
        num_workers: Dataloader worker processes.
        seed: Seed for every stochastic step (channel subsets, coresets, splits).
        epochs: Training epochs for methods that need gradient descent.
        severstal_target_prevalence: If set, downsample Severstal's test split
            so its anomalous-image rate matches this fraction, making it
            comparable to the other datasets' prevalence. ``None`` (default)
            leaves the test split untouched.
    """

    device: str = "cuda"
    max_fit_images: int = 500
    max_test_images: int = 0
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 0
    epochs: int = 0
    severstal_target_prevalence: float | None = None


class Scorer(Protocol):
    """A fitted model that can turn image paths into anomaly scores."""

    def score(self, paths: list[Path]) -> np.ndarray:
        """Return one anomaly score per path, higher meaning more anomalous."""
        ...


def scorer_labels(scorer: Scorer) -> np.ndarray | None:
    """Return a scorer's own ground-truth ordering, if it imposes one.

    Native scorers accept an arbitrary path list and return scores in that
    order, so the runner's own labels line up. Delegated backends may instead
    own their evaluation loop and return scores in an order the runner did not
    choose; those expose a ``labels`` attribute alongside the scores, and
    ignoring it would silently misalign every prediction.
    """
    labels = getattr(scorer, "labels", None)
    return None if labels is None else np.asarray(labels).astype(bool)


#: Builds a fitted scorer from normal-only training images.
#:
#: The labeled test split is passed in as well. Training-free methods ignore
#: it -- they must never look at it, and none of them do -- but backends that
#: own their own fit/evaluate loop need the split up front to construct it.
FitFn = Callable[[DatasetConfig, list[Path], list[tuple[Path, bool]], RunOptions], Scorer]


@dataclass(frozen=True)
class MethodSpec:
    """One entry in the zoo.

    Attributes:
        name: Unique key, e.g. ``patchcore_wide_resnet50_2``.
        family: Grouping used in the leaderboard (``padim``, ``patchcore``,
            ``anomalib``, ...).
        backend: ``"native"`` for the implementations in this package,
            ``"anomalib"`` for delegated models.
        fit: Callable that fits and returns a :class:`Scorer`.
        exportable: Whether a fitted instance can be written as a single ONNX
            graph matching the production ``(3, H, W) -> scalar`` contract.
        notes: Short human-readable description for the report.
        tags: Free-form labels used by ``--models`` selectors.
        trainable: Whether fitting this method involves gradient descent.
            Training-free methods (memory banks, Gaussian fits) leave this
            ``False``; their ``training_wall_clock_seconds`` is reported as 0
            and they remain eligible for the few-shot pass.
        allowed_regimes: Regimes this method may run under, or ``None`` for no
            restriction. An explicit, narrow override -- today used only to
            keep Dinomaly out of the one-class regime (see methods_dinomaly.py).
    """

    name: str
    family: str
    backend: str
    fit: FitFn
    exportable: bool = False
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    trainable: bool = False
    allowed_regimes: tuple[str, ...] | None = None


_REGISTRY: dict[str, MethodSpec] = {}


def register(spec: MethodSpec) -> MethodSpec:
    """Add ``spec`` to the global registry.

    Raises:
        ValueError: If the name is already taken.
    """
    if spec.name in _REGISTRY:
        msg = f"Duplicate method name: {spec.name!r}"
        raise ValueError(msg)
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> MethodSpec:
    """Look up one method by name.

    Raises:
        SystemExit: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        msg = f"Unknown method {name!r}. Run with --list to see the zoo."
        raise SystemExit(msg)
    return _REGISTRY[name]


def all_methods() -> list[MethodSpec]:
    """Every registered method, sorted by name."""
    return sorted(_REGISTRY.values(), key=lambda spec: spec.name)


def select(selectors: list[str]) -> list[MethodSpec]:
    """Resolve ``--models`` selectors into concrete methods.

    A selector is a method name, a family name, a backend name, a tag, or
    ``all``. Selectors accumulate, so ``--models patchcore padim`` runs both
    families and ``--models all`` runs the zoo.

    Args:
        selectors: The selector strings.

    Returns:
        Matching methods, de-duplicated and sorted by name.

    Raises:
        SystemExit: If a selector matches nothing.
    """
    chosen: dict[str, MethodSpec] = {}
    for selector in selectors:
        if selector == "all":
            chosen.update({spec.name: spec for spec in all_methods()})
            continue
        matches = [
            spec
            for spec in all_methods()
            if selector in (spec.name, spec.family, spec.backend) or selector in spec.tags
        ]
        if not matches:
            msg = f"Selector {selector!r} matched no method. Run with --list to see the zoo."
            raise SystemExit(msg)
        chosen.update({spec.name: spec for spec in matches})
    return sorted(chosen.values(), key=lambda spec: spec.name)
