"""The model zoo: backbones, every scoring method, and one factory to get any of them.

Every entry -- whether a training-free memory bank implemented natively or a
gradient-trained model delegated to Anomalib -- exposes the same two steps:
fit on normal-only images, then score a labeled test split. Callers never
need to know which family they're driving, which is what keeps comparisons
across the zoo apples-to-apples.

Organized bottom-up:

1. :class:`BackboneSpec` / :class:`FrozenFeatureExtractor` -- the swappable,
   frozen pretrained trunks every native method scores on top of.
2. The method registry (:class:`MethodSpec`, :func:`register`, :func:`get`,
   :func:`select`) -- one uniform interface over the whole zoo.
3. The training-free native methods: PaDiM, PatchCore, DFM.
4. The Dinomaly reconstruction network (encoder/bottleneck/decoder).
5. Dinomaly's own training loop and registration -- the one method here that
   needs gradient descent.
6. The optional Anomalib adapter, for the ~25 gradient-trained methods this
   project does not reimplement.
7. :func:`get_model`, a single convenience factory over the whole registry.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import timm
import torch
from torch import nn

from training.data import DatasetConfig, image_loader

try:  # pragma: no cover - exercised only when the optional dep is present
    import anomalib.models as anomalib_models
    from anomalib.data import Folder
    from anomalib.engine import Engine

    ANOMALIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANOMALIB_AVAILABLE = False

# -----------------------------------------------------------------------------
# Backbones: frozen pretrained feature extractors shared by every native method
# -----------------------------------------------------------------------------
#
# The single biggest lever on embedding-based anomaly detection is not the
# scoring rule but the features it scores. Turning the backbone into a
# swappable axis means PaDiM-on-ResNet18 and PatchCore-on-DINOv2 are the same
# experiment with two knobs changed rather than two separate codebases.
#
# Everything here is plain tensor ops on a frozen trunk -- no gradients, no
# training -- so a scorer fitted on top of one of these extractors still
# exports to a single ONNX graph matching the production contract.

#: ImageNet statistics; every backbone in the registry was pretrained with them.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneSpec:
    """Declarative description of one pretrained trunk.

    Attributes:
        name: Short key used in method names and result rows.
        timm_name: Model identifier passed to ``timm.create_model``.
        kind: ``"cnn"`` for hierarchical conv trunks, ``"vit"`` for transformers.
        layers: Stages concatenated by default, i.e. the PatchCore-equivalent
            mid-level pair (ResNet ``layer2``+``layer3`` and its analogue in
            other trunks). For CNNs these are ``features_only`` out-indices,
            which are *not* the torchvision ``layerN`` numbers -- timm counts
            the stem as index 0, so ResNet ``layer2``+``layer3`` is ``(2, 3)``.
            For ViTs they are block indices, negative counting back from the
            last block.
        padim_layers: Stages for PaDiM, which concatenates three scales
            starting one level finer than :attr:`layers`.
        patch_multiple: Input side lengths are rounded to a multiple of this
            before the trunk sees them (ViT patch grids require it).
        input_scale: Factor applied to the input before the trunk sees it.
            A patch-14 ViT at 256px yields an 18x18 grid against the CNNs'
            32x32, and that coarser grid -- not the features -- would be what
            loses small defects. Upscaling the ViT input restores a comparable
            patch density so the sweep measures the features, not the stride.
        vram_heavy: Hint that this trunk needs a reduced batch size.
    """

    name: str
    timm_name: str
    kind: str
    layers: tuple[int, ...]
    padim_layers: tuple[int, ...]
    patch_multiple: int = 1
    input_scale: float = 1.0
    vram_heavy: bool = False


#: The backbone axis of the sweep. ResNet/WideResNet are the literature
#: baselines (PaDiM, PatchCore); ConvNeXt and EfficientNet test whether a
#: stronger supervised trunk transfers; the DINOv2 ViTs test self-supervised
#: features, which is where recent work (AnomalyDINO, Dinomaly, SuperADD)
#: draws most of its gain.
BACKBONES: dict[str, BackboneSpec] = {
    spec.name: spec
    for spec in (
        BackboneSpec("resnet18", "resnet18", "cnn", (2, 3), (1, 2, 3)),
        BackboneSpec("resnet50", "resnet50", "cnn", (2, 3), (1, 2, 3)),
        BackboneSpec("wide_resnet50_2", "wide_resnet50_2", "cnn", (2, 3), (1, 2, 3)),
        BackboneSpec("convnext_small", "convnext_small.fb_in22k_ft_in1k", "cnn", (1, 2), (0, 1, 2)),
        BackboneSpec("efficientnet_b4", "tf_efficientnet_b4.ns_jft_in1k", "cnn", (2, 3), (1, 2, 3)),
        BackboneSpec(
            "dinov2_vits14",
            "vit_small_patch14_dinov2.lvd142m",
            "vit",
            (-4, -1),
            (-7, -4, -1),
            14,
            2.0,
        ),
        BackboneSpec(
            "dinov2_vitb14",
            "vit_base_patch14_dinov2.lvd142m",
            "vit",
            (-4, -1),
            (-7, -4, -1),
            14,
            2.0,
            True,
        ),
        BackboneSpec(
            "dinov2_vitl14",
            "vit_large_patch14_dinov2.lvd142m",
            "vit",
            (-8, -1),
            (-12, -6, -1),
            14,
            2.0,
            True,
        ),
    )
}


class FrozenFeatureExtractor(nn.Module):
    """Concatenated multi-scale features from a frozen pretrained trunk.

    The forward contract matches the rest of the training tree: input is a
    ``(N, 3, H, W)`` float tensor with values in ``[0, 255]``, output is a
    ``(N, C, h, w)`` feature map where every contributing stage has been
    resized onto the finest stage's grid.

    Args:
        spec: The backbone to instantiate.
        layers: Stage override; defaults to ``spec.layers``.
    """

    def __init__(self, spec: BackboneSpec, layers: tuple[int, ...] | None = None) -> None:
        """Build and freeze the trunk."""
        super().__init__()
        self.spec = spec
        self.layers = layers if layers is not None else spec.layers

        if spec.kind == "cnn":
            self.trunk = timm.create_model(
                spec.timm_name,
                pretrained=True,
                features_only=True,
                out_indices=self.layers,
            )
            channels = tuple(self.trunk.feature_info.channels())
        else:
            self.trunk = timm.create_model(
                spec.timm_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
            )
            channels = (self.trunk.embed_dim,) * len(self.layers)

        self._num_channels = int(sum(channels))
        self.trunk.eval()
        for param in self.trunk.parameters():
            param.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def num_channels(self) -> int:
        """Total channel count of the concatenated feature map."""
        return self._num_channels

    def grid_size(self, height: int, width: int) -> tuple[int, int]:
        """Feature-grid size produced for an input of ``(height, width)``."""
        with torch.no_grad():
            probe = torch.zeros(1, 3, height, width, device=self.mean.device)
            shape = self(probe).shape
        return int(shape[-2]), int(shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract concatenated features from a ``(N, 3, H, W)`` batch in ``[0, 255]``."""
        x = (x / 255.0 - self.mean) / self.std
        x = self._fit_patch_grid(x)

        if self.spec.kind == "cnn":
            features = list(self.trunk(x))
        else:
            features = list(
                self.trunk.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
            )

        size = features[0].shape[-2:]
        resized = [
            f
            if f.shape[-2:] == size
            else nn.functional.interpolate(f, size=size, mode="bilinear", align_corners=False)
            for f in features
        ]
        return torch.cat(resized, dim=1)

    def _fit_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the trunk's input scale and round to its patch multiple."""
        multiple = self.spec.patch_multiple
        scale = self.spec.input_scale
        if multiple <= 1 and scale == 1.0:
            return x
        height, width = int(x.shape[-2]), int(x.shape[-1])
        target_h = max(multiple, round(height * scale / multiple) * multiple)
        target_w = max(multiple, round(width * scale / multiple) * multiple)
        if (target_h, target_w) == (height, width):
            return x
        return nn.functional.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )


def build_extractor(
    backbone: str, device: torch.device, layers: tuple[int, ...] | None = None
) -> FrozenFeatureExtractor:
    """Instantiate a frozen extractor on ``device``.

    Args:
        backbone: Key into :data:`BACKBONES`.
        device: Device to place the trunk on.
        layers: Optional stage override.

    Returns:
        The extractor in eval mode with gradients disabled.

    Raises:
        SystemExit: If ``backbone`` is not a known key.
    """
    if backbone not in BACKBONES:
        msg = f"Unknown backbone {backbone!r}; known: {sorted(BACKBONES)}"
        raise SystemExit(msg)
    extractor = FrozenFeatureExtractor(BACKBONES[backbone], layers).to(device)
    extractor.eval()
    return extractor


# -----------------------------------------------------------------------------
# The method registry: one uniform interface over the whole zoo
# -----------------------------------------------------------------------------


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
    order, so the caller's own labels line up. Delegated backends may instead
    own their evaluation loop and return scores in an order the caller did not
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
        backend: ``"native"`` for the implementations in this module,
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
            keep Dinomaly out of the one-class regime (see
            :func:`_register_dinomaly_methods`).
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
    """Look up one method by its exact registered name.

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


# -----------------------------------------------------------------------------
# Native methods: PaDiM, PatchCore, DFM -- all training-free
# -----------------------------------------------------------------------------
#
# These need no gradient descent at all: fitting is one forward pass over the
# normal images plus some linear algebra. That matters for a production line
# -- there is no learning rate to tune, no epoch count to babysit, and
# re-fitting after a process change is minutes rather than hours -- and it
# keeps every one of them exportable as a single ONNX graph.
#
# * PaDiM -- a Gaussian per patch position, scored by Mahalanobis distance.
#   Strong when the part is camera-aligned, since position (i, j) then means
#   the same thing in every image.
# * PatchCore -- a coreset-subsampled memory bank of normal patches, scored
#   by nearest-neighbour distance. Position-agnostic, so it handles unaligned
#   material (steel strip) that PaDiM's per-position prior gets wrong.
# * DFM -- a PCA subspace over globally pooled features, scored by
#   reconstruction error. Much cheaper than either, and a useful control:
#   when it wins, the defect signal was global and the patch machinery
#   bought nothing.

#: Patch features are projected to this many dimensions before they are
#: stored or compared. Full WideResNet/DINOv2 features are 1024-2048 channels
#: wide; the papers all reduce, and on an 8 GB card it is the difference
#: between fitting and not.
_PADIM_DIMS = 100
_PATCHCORE_DIMS = 384

#: Upper bound on patches held in memory while building a PatchCore bank.
#: Severstal alone would otherwise pool ~1.6M patches.
MAX_POOL_PATCHES = 250_000


def _resolve_batch_size(backbone: str, options: RunOptions) -> int:
    """Halve the batch for VRAM-heavy trunks so 8 GB cards survive the sweep."""
    if BACKBONES[backbone].vram_heavy:
        return max(1, options.batch_size // 4)
    return options.batch_size


def _local_aggregate(feature_map: torch.Tensor, kernel: int) -> torch.Tensor:
    """Average-pool each patch with its neighbours, preserving the grid size.

    PatchCore calls this "locally aware" patch features: it widens each
    patch's receptive field without moving to a coarser, lower-resolution
    stage, which is what lets a small defect stay localized.

    ``count_include_pad=False`` matters more than it looks. With the default,
    border patches are averaged over a window that is one third zeros, so
    every border feature is scaled down relative to the interior ones. The
    bank then has no near neighbour for a normal border patch and the whole
    frame edge lights up as anomalous.
    """
    if kernel <= 1:
        return feature_map
    return nn.functional.avg_pool2d(
        feature_map,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
        count_include_pad=False,
    )


class _Projection(nn.Module):
    """Fixed, seeded dimensionality reduction applied to every patch vector.

    Two modes, matching what the two method families actually do: PaDiM keeps
    a random *subset* of channels, PatchCore applies a random *projection*.
    Both are frozen at fit time, so the reduction is part of the exported
    graph rather than a preprocessing step the caller has to reproduce.

    Args:
        in_channels: Channel count of the incoming feature map.
        out_dims: Target dimensionality.
        mode: ``"subset"`` or ``"project"``.
        seed: Seed controlling the subset/matrix draw.
    """

    def __init__(self, in_channels: int, out_dims: int, mode: str, seed: int) -> None:
        """Draw and freeze the reduction."""
        super().__init__()
        self.mode = mode if out_dims < in_channels else "none"
        self.out_dims = min(out_dims, in_channels)
        generator = torch.Generator().manual_seed(seed)
        if self.mode == "subset":
            indices = torch.randperm(in_channels, generator=generator)[: self.out_dims]
            self.register_buffer("indices", indices.sort().values)
        elif self.mode == "project":
            matrix = torch.randn(in_channels, self.out_dims, generator=generator)
            self.register_buffer("matrix", matrix / (self.out_dims**0.5))
        else:
            self.out_dims = in_channels

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Reduce a ``(N, P, C)`` patch tensor to ``(N, P, d)``."""
        if self.mode == "subset":
            return torch.index_select(patches, 2, self.indices)
        if self.mode == "project":
            return patches @ self.matrix
        return patches


class EmbeddingScorer(nn.Module):
    """Shared machinery: turn image paths into per-image anomaly scores.

    Subclasses implement :meth:`score_patches`; everything else -- batching,
    feature extraction, local aggregation, projection -- is common, which is
    precisely what makes the resulting numbers comparable across methods.

    This is an ``nn.Module`` rather than a plain object so that everything a
    fitted method learned (a Gaussian's inverse covariance, a memory bank, a
    PCA basis) lives in registered buffers. That is what lets
    ``training.export`` trace the whole scorer -- backbone included -- into
    one ONNX graph instead of shipping a model plus a pickle of side state.

    Args:
        extractor: Frozen backbone.
        projection: Fitted dimensionality reduction.
        config: Dataset configuration supplying the input geometry.
        options: Sweep-wide knobs.
        aggregate_kernel: Local-aggregation window; ``1`` disables it.
        batch_size: Images per forward pass.
        score_top_ratio: Fraction of highest-scoring patches averaged into the
            image score. ``0`` takes the maximum, as the papers specify.
    """

    #: Whether this method yields a spatial heatmap. Methods that pool away
    #: all spatial structure before scoring set this to False.
    produces_maps: bool = True

    def __init__(
        self,
        extractor: nn.Module,
        projection: _Projection,
        config: DatasetConfig,
        options: RunOptions,
        aggregate_kernel: int,
        batch_size: int,
        score_top_ratio: float = 0.0,
    ) -> None:
        """Store the fitted pieces."""
        super().__init__()
        self.extractor = extractor
        self.projection = projection
        self.config = config
        self.options = options
        self.aggregate_kernel = aggregate_kernel
        self.batch_size = batch_size
        self.score_top_ratio = score_top_ratio
        self.device = torch.device(options.device)
        self.grid_h = 0
        self.grid_w = 0

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Extract reduced patch features ``(N, P, d)`` from an image batch.

        The feature-grid shape is recorded on the instance as a side effect so
        :meth:`patch_map` can fold a flat ``(N, P)`` score vector back into a
        spatial heatmap. It is stored as plain ints, not tensors, so it traces
        as a constant and does not disturb ONNX export.
        """
        feature_map = self.extractor(images)
        feature_map = _local_aggregate(feature_map, self.aggregate_kernel)
        n, channels, height, width = feature_map.shape
        self.grid_h, self.grid_w = int(height), int(width)
        patches = feature_map.reshape(n, channels, height * width).transpose(1, 2)
        return self.projection(patches)

    def patch_scores(self, patches: torch.Tensor) -> torch.Tensor:
        """Reduce ``(N, P, d)`` patch features to a ``(N, P)`` per-patch score."""
        raise NotImplementedError

    def score_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Pool per-patch scores into one ``(N,)`` image-level score.

        Defaults to the maximum, which is what PaDiM and PatchCore both
        specify. ``score_top_ratio`` switches to the mean of the top fraction
        of patches instead: steadier on textures where a defect covers many
        patches, and less hostage to a single hot pixel, at the cost of
        blunting genuinely tiny defects.
        """
        return self._pool(self.patch_scores(patches))

    def patch_map(self, patches: torch.Tensor) -> torch.Tensor:
        """Per-patch scores folded back into a ``(N, h, w)`` spatial map."""
        per_patch = self.patch_scores(patches)
        return per_patch.reshape(per_patch.shape[0], self.grid_h, self.grid_w)

    def score(self, paths: list[Path]) -> np.ndarray:
        """Score every path in order, higher meaning more anomalous."""
        scores, _ = self.score_with_maps(list(paths), want_maps=False)
        return scores

    def score_with_maps(
        self, paths: list[Path], want_maps: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Score every path, optionally also returning full-resolution heatmaps.

        Args:
            paths: Images to score, in order.
            want_maps: Whether to also build anomaly maps. Skipping them
                avoids the upsample and the host transfer, which is most of
                the cost on the larger splits.

        Returns:
            ``(scores, maps)`` where ``maps`` is ``(N, H, W)`` at the config's
            input geometry, or ``None`` when maps were not requested or this
            method does not produce them.
        """
        loader = image_loader(
            list(paths),
            self.config.height,
            self.config.width,
            self.batch_size,
            self.options.num_workers,
        )
        emit_maps = want_maps and self.produces_maps
        scores: list[torch.Tensor] = []
        maps: list[torch.Tensor] = []

        with torch.no_grad():
            for images in loader:
                images = images.to(self.device, non_blocking=True)
                patches = self.embed(images)
                per_patch = self.patch_scores(patches)
                scores.append(self._pool(per_patch).float().cpu())
                if emit_maps:
                    grid = per_patch.reshape(per_patch.shape[0], 1, self.grid_h, self.grid_w)
                    upsampled = nn.functional.interpolate(
                        grid.float(),
                        size=(self.config.height, self.config.width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    maps.append(upsampled.squeeze(1).cpu())

        if not scores:
            return np.empty(0), None
        stacked = torch.cat(scores).numpy()
        return stacked, (torch.cat(maps).numpy() if maps else None)

    def _pool(self, per_patch: torch.Tensor) -> torch.Tensor:
        """Pool ``(N, P)`` per-patch scores to ``(N,)``; see :meth:`score_patches`."""
        if self.score_top_ratio <= 0.0:
            return per_patch.max(dim=1).values
        k = max(1, int(per_patch.shape[1] * self.score_top_ratio))
        return per_patch.topk(k, dim=1).values.mean(dim=1)

    def iter_fit_patches(self, paths: list[Path]) -> Iterator[torch.Tensor]:
        """Yield reduced patch features for the fit set, one batch at a time."""
        loader = image_loader(
            list(paths),
            self.config.height,
            self.config.width,
            self.batch_size,
            self.options.num_workers,
        )
        with torch.no_grad():
            for images in loader:
                yield self.embed(images.to(self.device, non_blocking=True))


class PaDiMScorer(EmbeddingScorer):
    """Mahalanobis distance to a Gaussian fitted per patch position.

    Args:
        pooled: Fit one position-agnostic Gaussian instead of one per
            position. Correct for unaligned material where patch coordinates
            carry no consistent meaning.
        eps: Diagonal regularization added before inversion.
    """

    def __init__(self, *args: object, pooled: bool, eps: float = 0.01, **kwargs: object) -> None:
        """Initialize with the Gaussian variant selected."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.pooled = pooled
        self.eps = eps
        self.register_buffer("mean", None)
        self.register_buffer("inv_cov", None)

    def fit(self, paths: list[Path]) -> PaDiMScorer:
        """Accumulate first and second moments over the fit set and invert.

        Moments are accumulated in a streaming fashion rather than by stacking
        every feature tensor: the stacked form is ``N x P x d`` floats, which
        for Severstal is tens of gigabytes, while the moments are a fixed
        ``P x d x d`` regardless of how many images are used.
        """
        count = 0
        sum_x: torch.Tensor | None = None
        sum_outer: torch.Tensor | None = None

        for patches in self.iter_fit_patches(paths):
            batch = patches.double()
            if self.pooled:
                # One position-agnostic Gaussian: pool every patch across the
                # whole batch into a flat (batch*positions, d) sample set
                # before accumulating. Keeping a "positions" axis sized off
                # batch*positions instead (the previous shape here) breaks
                # the moment a later batch has a different image count --
                # e.g. iter_fit_patches' final, shorter batch -- since the
                # accumulator's shape is fixed from the first batch seen,
                # and `.sum(dim=0)` over a leading singleton axis is a
                # no-op, not the reduction it needs to be.
                batch = batch.reshape(-1, batch.shape[-1])
                if sum_x is None:
                    dims = batch.shape[-1]
                    sum_x = torch.zeros(1, dims, dtype=torch.float64, device=self.device)
                    sum_outer = torch.zeros(1, dims, dims, dtype=torch.float64, device=self.device)
                sum_x += batch.sum(dim=0, keepdim=True)
                sum_outer += torch.einsum("ni,nj->ij", batch, batch).unsqueeze(0)
                count += batch.shape[0]
            else:
                if sum_x is None:
                    positions, dims = batch.shape[1], batch.shape[2]
                    sum_x = torch.zeros(positions, dims, dtype=torch.float64, device=self.device)
                    sum_outer = torch.zeros(
                        positions, dims, dims, dtype=torch.float64, device=self.device
                    )
                sum_x += batch.sum(dim=0)
                sum_outer += torch.einsum("npi,npj->pij", batch, batch)
                count += batch.shape[0]

        if sum_x is None or sum_outer is None or count < 2:
            msg = "PaDiM needs at least two fit batches worth of images"
            raise SystemExit(msg)

        mean = sum_x / count
        cov = (sum_outer - count * torch.einsum("pi,pj->pij", mean, mean)) / (count - 1)
        identity = torch.eye(cov.shape[-1], dtype=torch.float64, device=self.device)
        cov += self.eps * identity
        self.inv_cov = _batched_inverse(cov).float()
        self.mean = mean.float()
        return self

    def patch_scores(self, patches: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance of each patch to its position's Gaussian."""
        assert self.mean is not None and self.inv_cov is not None
        diff = patches - self.mean.unsqueeze(0)
        weighted = torch.einsum("npi,pij->npj", diff, self.inv_cov)
        return (weighted * diff).sum(dim=-1).clamp_min(0).sqrt()


class PatchCoreScorer(EmbeddingScorer):
    """Nearest-neighbour distance to a coreset-subsampled bank of normal patches.

    Args:
        coreset_ratio: Fraction of pooled patches retained in the bank.
        max_bank: Hard cap on bank size, so scoring latency stays bounded.
        neighbors: Number of nearest bank entries averaged per patch.
    """

    def __init__(
        self,
        *args: object,
        coreset_ratio: float = 0.01,
        max_bank: int = 20_000,
        neighbors: int = 1,
        **kwargs: object,
    ) -> None:
        """Initialize with the bank-construction parameters."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.coreset_ratio = coreset_ratio
        self.max_bank = max_bank
        self.neighbors = neighbors
        self.register_buffer("bank", None)

    def fit(self, paths: list[Path]) -> PatchCoreScorer:
        """Pool normal patches, subsample to a coreset, and keep it as the bank."""
        generator = torch.Generator(device="cpu").manual_seed(self.options.seed)
        pool: list[torch.Tensor] = []
        pooled_count = 0
        per_image_budget: int | None = None

        for patches in self.iter_fit_patches(paths):
            flat = patches.reshape(-1, patches.shape[-1]).half().cpu()
            # Reservoir-style cap: once the pool is large enough, thin each
            # incoming batch instead of growing without bound.
            if per_image_budget is None:
                estimated = max(1, len(paths)) * patches.shape[1]
                if estimated > MAX_POOL_PATCHES:
                    per_image_budget = max(1, MAX_POOL_PATCHES // max(1, len(paths)))
            if per_image_budget is not None:
                keep = min(flat.shape[0], per_image_budget * patches.shape[0])
                idx = torch.randperm(flat.shape[0], generator=generator)[:keep]
                flat = flat[idx]
            pool.append(flat)
            pooled_count += flat.shape[0]

        if not pool:
            msg = "PatchCore found no fit images"
            raise SystemExit(msg)

        pooled = torch.cat(pool)
        target = int(min(self.max_bank, max(16, round(pooled.shape[0] * self.coreset_ratio))))
        target = min(target, pooled.shape[0])
        indices = greedy_coreset(pooled.to(self.device).float(), target, self.options.seed)
        self.bank = pooled[indices.cpu()].to(self.device).float()
        return self

    def patch_scores(self, patches: torch.Tensor) -> torch.Tensor:
        """Mean distance from each patch to its ``k`` nearest bank entries."""
        assert self.bank is not None
        n, positions, dims = patches.shape
        flat = patches.reshape(n * positions, dims)
        distance = torch.cdist(flat, self.bank)
        k = min(self.neighbors, self.bank.shape[0])
        nearest = distance.topk(k, dim=1, largest=False).values.mean(dim=1)
        return nearest.reshape(n, positions)


class DFMScorer(EmbeddingScorer):
    """PCA reconstruction error over globally pooled features.

    A deliberately cheap control. It throws away all spatial information, so
    beating the patch-based methods here means the defect changes the image
    globally -- useful to know before paying for a memory bank on every frame.

    Args:
        variance: Fraction of feature variance the retained subspace must cover.
    """

    #: Spatial structure is averaged away before scoring, so there is no
    #: meaningful heatmap to draw -- a uniform one would be a lie.
    produces_maps: bool = False

    def __init__(self, *args: object, variance: float = 0.97, **kwargs: object) -> None:
        """Initialize with the retained-variance target."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.variance = variance
        self.register_buffer("mean", None)
        self.register_buffer("components", None)

    def fit(self, paths: list[Path]) -> DFMScorer:
        """Fit the PCA subspace on globally pooled normal features."""
        vectors = [patches.mean(dim=1).double().cpu() for patches in self.iter_fit_patches(paths)]
        if not vectors:
            msg = "DFM found no fit images"
            raise SystemExit(msg)
        matrix = torch.cat(vectors)
        mean = matrix.mean(dim=0)
        centered = matrix - mean
        _, singular, right = torch.linalg.svd(centered, full_matrices=False)
        energy = torch.cumsum(singular**2, dim=0) / torch.clamp((singular**2).sum(), min=1e-12)
        keep = int(torch.searchsorted(energy, torch.tensor(self.variance)).item()) + 1
        keep = max(1, min(keep, right.shape[0]))
        self.mean = mean.float().to(self.device)
        self.components = right[:keep].float().to(self.device)
        return self

    def patch_scores(self, patches: torch.Tensor) -> torch.Tensor:
        """Broadcast the single image-level residual across every patch slot.

        DFM pools before scoring, so there is one score per image, not per
        patch. Broadcasting keeps the ``(N, P)`` contract the base class
        expects; ``produces_maps`` is False so nothing renders it as a heatmap.
        """
        assert self.mean is not None and self.components is not None
        pooled = patches.mean(dim=1) - self.mean
        projected = pooled @ self.components.T @ self.components
        residual = (pooled - projected).norm(dim=1)
        return residual.unsqueeze(1).expand(-1, patches.shape[1])


def _batched_inverse(cov: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Invert a stack of covariance matrices in chunks to bound peak memory."""
    parts = [
        torch.linalg.inv(cov[start : start + chunk]) for start in range(0, cov.shape[0], chunk)
    ]
    return torch.cat(parts)


def greedy_coreset(features: torch.Tensor, target: int, seed: int) -> torch.Tensor:
    """Select ``target`` maximally-spread rows by k-center-greedy.

    Random subsampling would keep the bank's *density* but lose its *extent*,
    and it is the extent that matters: a normal patch type seen only a few
    times still has to be represented, or every instance of it scores as an
    anomaly at inference.

    Args:
        features: ``(N, d)`` candidate rows on the target device.
        target: Number of rows to keep.
        seed: Seed for the starting row.

    Returns:
        A ``(target,)`` long tensor of selected row indices.
    """
    n = features.shape[0]
    if target >= n:
        return torch.arange(n, device=features.device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = int(torch.randint(n, (1,), generator=generator).item())

    selected = torch.empty(target, dtype=torch.long, device=features.device)
    selected[0] = start
    min_distance = torch.cdist(features, features[start : start + 1]).squeeze(1)

    for step in range(1, target):
        nxt = int(torch.argmax(min_distance).item())
        selected[step] = nxt
        distance = torch.cdist(features, features[nxt : nxt + 1]).squeeze(1)
        min_distance = torch.minimum(min_distance, distance)
    return selected


def _build_common(
    config: DatasetConfig,
    options: RunOptions,
    backbone: str,
    layers: tuple[int, ...] | None,
    dims: int,
    mode: str,
) -> tuple[nn.Module, _Projection, int]:
    """Instantiate the trunk plus a seeded projection sized to its output."""
    device = torch.device(options.device)
    extractor = build_extractor(backbone, device, layers)
    projection = _Projection(extractor.num_channels, dims, mode, options.seed).to(device)
    return extractor, projection, _resolve_batch_size(backbone, options)


def _fit_padim(backbone: str, pooled: bool | None):
    """Build a fit function for PaDiM on ``backbone``."""

    def fit(
        config: DatasetConfig,
        paths: list[Path],
        test_split: list[tuple[Path, bool]],
        options: RunOptions,
    ) -> PaDiMScorer:
        _ = test_split  # Training-free: the fit never sees a label.
        spec = BACKBONES[backbone]
        extractor, projection, batch = _build_common(
            config, options, backbone, spec.padim_layers, _PADIM_DIMS, "subset"
        )
        use_pooled = (not config.position_aligned) if pooled is None else pooled
        scorer = PaDiMScorer(
            extractor,
            projection,
            config,
            options,
            1,
            batch,
            pooled=use_pooled,
        )
        return scorer.fit(paths)

    return fit


def _fit_patchcore(backbone: str, coreset_ratio: float, neighbors: int):
    """Build a fit function for PatchCore on ``backbone``."""

    def fit(
        config: DatasetConfig,
        paths: list[Path],
        test_split: list[tuple[Path, bool]],
        options: RunOptions,
    ) -> PatchCoreScorer:
        _ = test_split  # Training-free: the fit never sees a label.
        extractor, projection, batch = _build_common(
            config, options, backbone, None, _PATCHCORE_DIMS, "project"
        )
        scorer = PatchCoreScorer(
            extractor,
            projection,
            config,
            options,
            3,
            batch,
            coreset_ratio=coreset_ratio,
            neighbors=neighbors,
        )
        return scorer.fit(paths)

    return fit


def _fit_dfm(backbone: str):
    """Build a fit function for DFM on ``backbone``."""

    def fit(
        config: DatasetConfig,
        paths: list[Path],
        test_split: list[tuple[Path, bool]],
        options: RunOptions,
    ) -> DFMScorer:
        _ = test_split  # Training-free: the fit never sees a label.
        extractor, projection, batch = _build_common(config, options, backbone, None, 4096, "none")
        scorer = DFMScorer(extractor, projection, config, options, 1, batch)
        return scorer.fit(paths)

    return fit


#: Backbones swept for the two headline families. Trimmed to one canonical
#: trunk for PaDiM (wide_resnet50_2) to cut compute waste; the position-
#: agnostic pooled variants below are kept as a real ablation regardless of
#: backbone. PaDiM's own multiclass/oneclass results (2026-08-27 sweep) put
#: it last across every backbone tried so far, so this trim stays - more
#: PaDiM backbones is low-expected-value compute, not a coverage gap.
#: PatchCore and DFM get the full registered backbone set: every CNN
#: depth/width point (resnet18/50, wide_resnet50_2), the stronger supervised
#: trunks (convnext_small, efficientnet_b4), and both practical DINOv2 ViT
#: scales (vitb14, vitl14 - vits14 is reserved for Dinomaly's own comparison).
_PADIM_BACKBONES = ("wide_resnet50_2",)
_BROAD_BACKBONES = (
    "resnet18",
    "resnet50",
    "wide_resnet50_2",
    "convnext_small",
    "efficientnet_b4",
    "dinov2_vitb14",
    "dinov2_vitl14",
)
_PATCHCORE_BACKBONES = _BROAD_BACKBONES


def _register_native_methods() -> None:
    """Populate the registry with every native method/backbone combination."""
    for backbone in _PADIM_BACKBONES:
        register(
            MethodSpec(
                name=f"padim_{backbone}",
                family="padim",
                backend="native",
                fit=_fit_padim(backbone, None),
                exportable=True,
                notes=f"PaDiM, per-position Gaussian, {backbone} trunk",
                tags=("training-free", "gaussian"),
            )
        )

    # Pooled PaDiM is the position-agnostic variant. On aligned datasets it
    # should lose to per-position PaDiM; on Severstal it should win. Running
    # both everywhere is how that claim gets tested rather than assumed.
    for backbone in ("resnet18", "wide_resnet50_2"):
        register(
            MethodSpec(
                name=f"padim_pooled_{backbone}",
                family="padim",
                backend="native",
                fit=_fit_padim(backbone, True),
                exportable=True,
                notes=f"PaDiM, single pooled Gaussian, {backbone} trunk",
                tags=("training-free", "gaussian", "position-agnostic"),
            )
        )

    for backbone in _PATCHCORE_BACKBONES:
        register(
            MethodSpec(
                name=f"patchcore_{backbone}",
                family="patchcore",
                backend="native",
                fit=_fit_patchcore(backbone, 0.01, 1),
                exportable=True,
                notes=f"PatchCore, 1% coreset memory bank, {backbone} trunk",
                tags=("training-free", "memory-bank", "position-agnostic"),
            )
        )

    # A denser bank costs latency and VRAM; whether it buys accuracy is
    # exactly the kind of thing the sweep exists to answer.
    register(
        MethodSpec(
            name="patchcore_dense_wide_resnet50_2",
            family="patchcore",
            backend="native",
            fit=_fit_patchcore("wide_resnet50_2", 0.10, 1),
            exportable=True,
            notes="PatchCore, 10% coreset memory bank, wide_resnet50_2 trunk",
            tags=("training-free", "memory-bank", "position-agnostic"),
        )
    )
    register(
        MethodSpec(
            name="patchcore_knn3_wide_resnet50_2",
            family="patchcore",
            backend="native",
            fit=_fit_patchcore("wide_resnet50_2", 0.01, 3),
            exportable=True,
            notes="PatchCore, 1% bank, 3-NN averaged distance, wide_resnet50_2 trunk",
            tags=("training-free", "memory-bank", "position-agnostic"),
        )
    )

    for backbone in _BROAD_BACKBONES:
        register(
            MethodSpec(
                name=f"dfm_{backbone}",
                family="dfm",
                backend="native",
                fit=_fit_dfm(backbone),
                exportable=True,
                notes=f"Deep Feature Modeling, PCA reconstruction error, {backbone} trunk",
                tags=("training-free", "baseline", "position-agnostic"),
            )
        )


# -----------------------------------------------------------------------------
# Dinomaly: the multi-class reconstruction model this benchmark is built around
# -----------------------------------------------------------------------------
#
# Every native method above fits one model per category. That is the setting
# the literature reports, and it is a poor fit for a real line: 29 categories
# means 29 checkpoints to version, deploy, calibrate and monitor, and a new
# product means a new model. Dinomaly (Guo et al., CVPR 2025) is the first
# method to close that gap -- one checkpoint covering every category at
# accuracy that matches per-category specialists.
#
# The design is deliberately plain: a frozen DINOv2 encoder, an MLP
# bottleneck, and a small Transformer decoder trained to reconstruct the
# encoder's features for normal images. Anomalies are wherever the
# reconstruction fails. The paper's contribution is four restraints that stop
# the decoder learning to reconstruct *everything*, which would leave nothing
# for the anomaly score to detect:
#
# * Frozen foundation encoder. DINOv2 features already separate normal from
#   abnormal; fine-tuning them destroys that.
# * Noisy bottleneck. Dropout at 0.2 on the bottleneck MLP injects noise, so
#   the decoder cannot memorize an identity mapping.
# * Linear attention. Its inability to focus sharply is the point -- a
#   softmax decoder attends precisely enough to copy anomalies through.
# * Loose reconstruction. Layers are fused into two groups rather than
#   matched one-to-one, and a hard-mining cosine loss down-weights the easy
#   points that would otherwise dominate the gradient.
#
# Implemented natively rather than pulled from Anomalib so it can be trained
# in the multi-class regime, exported, and cross-checked against Anomalib's
# own implementation -- two independent implementations agreeing is worth
# more than one number.

#: Encoder blocks tapped for features, and how they fuse into groups. Early
#: blocks carry texture, late blocks carry semantics; two groups keeps both
#: without imposing a strict layer-to-layer correspondence.
_TARGET_LAYERS = (2, 3, 4, 5, 6, 7, 8, 9)
_FUSE_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7))

#: Dropout on the bottleneck MLP. This *is* the noisy bottleneck.
_BOTTLENECK_DROPOUT = 0.2

#: Gaussian smoothing applied to the anomaly map before scoring.
_SMOOTH_KERNEL = 5
_SMOOTH_SIGMA = 4.0

#: Backbones, keyed by the short name used in method registration.
DINOMALY_BACKBONES: dict[str, str] = {
    "vits14": "vit_small_patch14_reg4_dinov2.lvd142m",
    "vitb14": "vit_base_patch14_reg4_dinov2.lvd142m",
    "vitl14": "vit_large_patch14_reg4_dinov2.lvd142m",
}


class LinearAttention(nn.Module):
    """Attention with an ELU feature map instead of softmax.

    Softmax attention can concentrate almost all its weight on one token,
    which lets the decoder copy an anomalous patch straight through and score
    it as normal. The ELU kernel spreads attention by construction, so the
    decoder reconstructs from context rather than from the token itself.

    Args:
        dim: Token dimensionality.
        num_heads: Attention heads.
        qkv_bias: Whether the qkv projection carries a bias.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        """Build the projections."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over ``(N, L, C)`` tokens."""
        n, length, channels = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # ELU + 1 keeps the feature map positive, which is what makes the
        # un-normalized product a valid attention kernel.
        query = nn.functional.elu(query) + 1.0
        key = nn.functional.elu(key) + 1.0

        # Associativity: (K^T V) first is O(L) in sequence length, not O(L^2).
        context = torch.einsum("nhld,nhle->nhde", key, value)
        normalizer = torch.einsum("nhld,nhd->nhl", query, key.sum(dim=2)).clamp_min(1e-6)
        out = torch.einsum("nhld,nhde->nhle", query, context) / normalizer.unsqueeze(-1)

        out = out.transpose(1, 2).reshape(n, length, channels)
        return self.proj(out)


class DecoderBlock(nn.Module):
    """Pre-norm Transformer block using :class:`LinearAttention`.

    Args:
        dim: Token dimensionality.
        num_heads: Attention heads.
        mlp_ratio: Hidden width of the MLP, as a multiple of ``dim``.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Build the block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP with residual connections."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Bottleneck(nn.Module):
    """The noisy bottleneck: one wide MLP with dropout on both stages.

    Args:
        dim: Token dimensionality.
        expansion: Hidden width multiplier.
        dropout: Dropout probability; this is the noise source.
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = _BOTTLENECK_DROPOUT) -> None:
        """Build the MLP."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project tokens through the noisy bottleneck."""
        return self.net(x)


class DinomalyModel(nn.Module):
    """Frozen DINOv2 encoder plus a trainable bottleneck and decoder.

    Args:
        backbone: Key into :data:`DINOMALY_BACKBONES`.
        decoder_depth: Number of decoder blocks.
    """

    def __init__(self, backbone: str = "vitb14", decoder_depth: int = 8) -> None:
        """Build the encoder, bottleneck and decoder."""
        super().__init__()
        if backbone not in DINOMALY_BACKBONES:
            msg = f"Unknown Dinomaly backbone {backbone!r}; known: {sorted(DINOMALY_BACKBONES)}"
            raise SystemExit(msg)

        self.encoder = timm.create_model(
            DINOMALY_BACKBONES[backbone],
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        dim = self.encoder.embed_dim
        num_heads = self.encoder.blocks[0].attn.num_heads
        self.embed_dim = dim
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.num_prefix_tokens = getattr(self.encoder, "num_prefix_tokens", 1)

        self.bottleneck = Bottleneck(dim)
        self.decoder = nn.ModuleList(DecoderBlock(dim, num_heads) for _ in range(decoder_depth))

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that receive gradients: the bottleneck and decoder only."""
        return list(self.bottleneck.parameters()) + list(self.decoder.parameters())

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], int, int]:
        """Run the frozen encoder and return the tapped block outputs.

        Args:
            x: ``(N, 3, H, W)`` batch with values in ``[0, 255]``.

        Returns:
            ``(features, grid_h, grid_w)`` where ``features`` holds one
            ``(N, L, C)`` tensor per tapped block, prefix tokens stripped.
        """
        x = (x / 255.0 - self.mean) / self.std
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size

        with torch.no_grad():
            tokens = self.encoder.patch_embed(x)
            tokens = self.encoder._pos_embed(tokens)
            tokens = self.encoder.norm_pre(tokens)
            taps: list[torch.Tensor] = []
            for index, block in enumerate(self.encoder.blocks):
                tokens = block(tokens)
                if index in _TARGET_LAYERS:
                    taps.append(tokens[:, self.num_prefix_tokens :])
        return taps, grid_h, grid_w

    @staticmethod
    def fuse(features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Sum tapped blocks into the two loose reconstruction groups."""
        return [torch.stack([features[i] for i in group]).sum(dim=0) for group in _FUSE_GROUPS]

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
        """Encode, reconstruct, and return both sides of the comparison.

        Returns:
            ``(encoder_groups, decoder_groups, grid_h, grid_w)``.
        """
        taps, grid_h, grid_w = self.encode(x)

        # The decoder consumes the *last* tapped block through the bottleneck
        # and is asked to regenerate the whole stack, which is what makes the
        # reconstruction loose rather than layer-to-layer.
        tokens = self.bottleneck(taps[-1])
        decoded: list[torch.Tensor] = []
        for block in self.decoder:
            tokens = block(tokens)
            decoded.append(tokens)

        return self.fuse(taps), self.fuse(decoded), grid_h, grid_w

    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Cosine-distance anomaly map at the input resolution.

        Args:
            x: ``(N, 3, H, W)`` batch with values in ``[0, 255]``.

        Returns:
            A ``(N, H, W)`` map, Gaussian-smoothed, higher meaning more anomalous.
        """
        encoder_groups, decoder_groups, grid_h, grid_w = self(x)
        height, width = int(x.shape[-2]), int(x.shape[-1])

        maps = []
        for encoded, decoded in zip(encoder_groups, decoder_groups, strict=True):
            distance = 1.0 - nn.functional.cosine_similarity(encoded, decoded, dim=-1)
            grid = distance.reshape(distance.shape[0], 1, grid_h, grid_w)
            maps.append(
                nn.functional.interpolate(
                    grid, size=(height, width), mode="bilinear", align_corners=False
                )
            )

        combined = torch.cat(maps, dim=1).mean(dim=1, keepdim=True)
        return _gaussian_blur(combined).squeeze(1)


def global_cosine_hard_mining(
    encoder_groups: list[torch.Tensor],
    decoder_groups: list[torch.Tensor],
    percentile: float,
    factor: float = 0.1,
) -> torch.Tensor:
    """Cosine reconstruction loss that down-weights already-easy points.

    Most tokens in a normal image are trivially reconstructable, and their
    gradients swamp the handful of genuinely difficult ones. Scaling the
    gradient of every point below the ``percentile`` cut by ``factor``
    concentrates training where it still has something to learn -- without
    which the decoder converges to a good average reconstruction and a useless
    anomaly score.

    Args:
        encoder_groups: Target features, one tensor per fused group.
        decoder_groups: Reconstructed features, aligned to the targets.
        percentile: Fraction of points treated as easy, ramped toward 0.9.
        factor: Gradient multiplier applied to those easy points.

    Returns:
        Scalar loss.
    """
    loss = torch.zeros((), device=encoder_groups[0].device)
    for encoded, decoded in zip(encoder_groups, decoder_groups, strict=True):
        target = encoded.detach()
        point_distance = 1.0 - nn.functional.cosine_similarity(target, decoded, dim=-1)
        loss = loss + point_distance.mean()

        if decoded.requires_grad and 0.0 < percentile < 1.0:
            keep = max(1, int(point_distance.numel() * (1.0 - percentile)))
            threshold = torch.topk(point_distance.reshape(-1), k=keep).values[-1]
            easy = (point_distance < threshold).unsqueeze(-1)
            decoded.register_hook(
                lambda grad, easy=easy, factor=factor: torch.where(easy, grad * factor, grad)
            )
    return loss / len(encoder_groups)


def _gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian smoothing of a ``(N, 1, H, W)`` map."""
    coords = torch.arange(_SMOOTH_KERNEL, dtype=x.dtype, device=x.device)
    coords = coords - (_SMOOTH_KERNEL - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2 * _SMOOTH_SIGMA**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    padding = _SMOOTH_KERNEL // 2

    x = nn.functional.conv2d(x, kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    return nn.functional.conv2d(x, kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))


class WarmCosineSchedule:
    """Linear warmup into cosine decay, matching the reference recipe.

    Args:
        optimizer: Optimizer whose learning rate is driven.
        base_lr: Peak learning rate reached after warmup.
        final_lr: Learning rate at the end of training.
        warmup_iters: Iterations spent ramping up.
        total_iters: Total training iterations.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        final_lr: float,
        warmup_iters: int,
        total_iters: int,
    ) -> None:
        """Store the schedule shape."""
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_iters = max(1, warmup_iters)
        self.total_iters = max(self.warmup_iters + 1, total_iters)
        self.step_count = 0

    def step(self) -> float:
        """Advance one iteration and apply the new learning rate."""
        if self.step_count < self.warmup_iters:
            lr = self.base_lr * (self.step_count + 1) / self.warmup_iters
        else:
            progress = (self.step_count - self.warmup_iters) / (
                self.total_iters - self.warmup_iters
            )
            progress = min(1.0, max(0.0, progress))
            lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step_count += 1
        return lr


# -----------------------------------------------------------------------------
# Dinomaly training and registration -- the one method here needing gradient descent
# -----------------------------------------------------------------------------
#
# Everything above this point either needs no training (the native methods)
# or is the frozen network definition. Dinomaly is the one method that owns
# its own training loop rather than pretending to fit the training-free
# mould -- but it exposes the same score/score_with_maps surface, so the
# sweep runner, the metrics, the ensembler and the dashboard treat it
# identically to every other scorer.
#
# The recipe follows the reference implementation: 448px resized to a 392px
# crop, StableAdamW-style AdamW at 2e-3 with amsgrad, warm-cosine decay to
# 2e-4, and a hard-mining percentile ramped to 0.9 over the first 1000
# iterations.

#: Reference training recipe.
_BASE_LR = 2e-3
_FINAL_LR = 2e-4
_WEIGHT_DECAY = 1e-4
_WARMUP_ITERS = 100
_DEFAULT_ITERS = 10_000

#: The hard-mining percentile ramps from 0 to this over ``_RAMP_ITERS``.
#: Starting at the target would starve early training of gradient entirely.
_TARGET_PERCENTILE = 0.9
_RAMP_ITERS = 1000

#: Dinomaly's own geometry. Patch-14 needs a multiple of 14, and the encoder
#: was pretrained at this scale, so the dataset-level sizes in
#: ``training.data`` are deliberately overridden here.
_INPUT_SIZE = 392

#: Fraction of highest-scoring pixels averaged into the image-level score.
_MAX_RATIO = 0.01


class DinomalyScorer(nn.Module):
    """A trained Dinomaly model exposing the benchmark's scorer surface.

    Args:
        model: The (already trained) Dinomaly network.
        options: Sweep-wide knobs.
        batch_size: Images per forward pass.
    """

    produces_maps: bool = True

    def __init__(self, model: DinomalyModel, options: RunOptions, batch_size: int) -> None:
        """Store the trained model."""
        super().__init__()
        self.model = model
        self.options = options
        self.batch_size = batch_size
        self.device = torch.device(options.device)
        # Dinomaly always runs at its own fixed, patch-14-aligned resolution
        # regardless of the dataset's declared geometry (see score_with_maps
        # below) -- exposed so a caller timing inference uses the real input
        # shape instead of the dataset's, which would otherwise either crash
        # (not a multiple of 14) or silently time the wrong size.
        self.input_size: tuple[int, int] = (_INPUT_SIZE, _INPUT_SIZE)

    def score(self, paths: list[Path]) -> np.ndarray:
        """Score every path in order, higher meaning more anomalous."""
        scores, _ = self.score_with_maps(list(paths), want_maps=False)
        return scores

    def score_with_maps(
        self, paths: list[Path], want_maps: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Score every path, optionally also returning heatmaps.

        Returns:
            ``(scores, maps)`` with maps at Dinomaly's own input resolution.
        """
        self.model.eval()
        loader = image_loader(
            list(paths), _INPUT_SIZE, _INPUT_SIZE, self.batch_size, self.options.num_workers
        )
        scores: list[torch.Tensor] = []
        maps: list[torch.Tensor] = []

        with torch.no_grad():
            for images in loader:
                anomaly_map = self.model.anomaly_map(images.to(self.device, non_blocking=True))
                scores.append(_pool_map(anomaly_map).float().cpu())
                if want_maps:
                    maps.append(anomaly_map.float().cpu())

        if not scores:
            return np.empty(0), None
        return torch.cat(scores).numpy(), (torch.cat(maps).numpy() if maps else None)


def _pool_map(anomaly_map: torch.Tensor) -> torch.Tensor:
    """Image score as the mean of the hottest :data:`_MAX_RATIO` of pixels.

    A pure maximum reads one pixel and is hostage to a single smoothing
    artifact; averaging the top 1% keeps the sensitivity to small defects
    while requiring the evidence to be more than one stray peak.
    """
    flat = anomaly_map.flatten(1)
    k = max(1, int(flat.shape[1] * _MAX_RATIO))
    return flat.topk(k, dim=1).values.mean(dim=1)


def train_dinomaly(
    paths: list[Path],
    options: RunOptions,
    backbone: str,
    iters: int,
    batch_size: int,
    verbose: bool = True,
) -> DinomalyScorer:
    """Train a Dinomaly model on normal-only images.

    Args:
        paths: Normal training images. In the multi-class regime this is the
            union across every category, which is the entire point of the method.
        options: Sweep-wide knobs.
        backbone: Key into :data:`DINOMALY_BACKBONES`.
        iters: Training iterations.
        batch_size: Images per step.
        verbose: Whether to print progress.

    Returns:
        The trained scorer.
    """
    device = torch.device(options.device)
    torch.manual_seed(options.seed)

    model = DinomalyModel(backbone).to(device)
    model.encoder.eval()

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=_BASE_LR,
        betas=(0.9, 0.999),
        weight_decay=_WEIGHT_DECAY,
        amsgrad=True,
    )
    schedule = WarmCosineSchedule(optimizer, _BASE_LR, _FINAL_LR, _WARMUP_ITERS, iters)

    loader = image_loader(list(paths), _INPUT_SIZE, _INPUT_SIZE, batch_size, options.num_workers)
    step = 0
    running = 0.0

    while step < iters:
        for images in loader:
            if step >= iters:
                break
            model.bottleneck.train()
            model.decoder.train()

            images = images.to(device, non_blocking=True)
            encoder_groups, decoder_groups, _, _ = model(images)
            percentile = _TARGET_PERCENTILE * min(1.0, step / _RAMP_ITERS)
            loss = global_cosine_hard_mining(encoder_groups, decoder_groups, percentile)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=0.1)
            optimizer.step()
            schedule.step()

            running += float(loss.detach())
            step += 1
            if verbose and step % 500 == 0:
                print(f"    iter {step}/{iters} loss={running / 500:.4f}", flush=True)
                running = 0.0

    return DinomalyScorer(model, options, batch_size)


def _resolve_iters(options: RunOptions, default: int) -> int:
    """Let ``--epochs`` stand in as an iteration override for this family."""
    return options.epochs if options.epochs > 0 else default


def _fit_dinomaly(backbone: str, iters: int):
    """Build a fit function for Dinomaly on ``backbone``."""

    def fit(
        config: DatasetConfig,
        paths: list[Path],
        test_split: list[tuple[Path, bool]],
        options: RunOptions,
    ) -> DinomalyScorer:
        _ = (config, test_split)  # Unsupervised: the fit never sees a label.
        batch_size = 8 if backbone == "vitl14" else 16
        batch_size = min(batch_size, max(1, options.batch_size))
        return train_dinomaly(
            paths,
            options,
            backbone,
            _resolve_iters(options, iters),
            batch_size,
        )

    return fit


def _register_dinomaly_methods() -> None:
    """Register one entry per Dinomaly backbone.

    Multi-class only: Guo et al., "Dinomaly: The Less Is More Philosophy in
    Multi-Class Unsupervised Anomaly Detection", CVPR 2025, Table 2, report a
    <0.2pp gap between one-class and multi-class Dinomaly on MVTec/VisA --
    within their own 5-seed noise band (+/-0.03). Running one-class here would
    just spend GPU time re-measuring noise, so it's cut from the sweep.
    """
    for backbone in DINOMALY_BACKBONES:
        register(
            MethodSpec(
                name=f"dinomaly_{backbone}",
                family="dinomaly",
                backend="native",
                fit=_fit_dinomaly(backbone, _DEFAULT_ITERS),
                exportable=False,
                notes=(
                    f"Dinomaly, frozen DINOv2-{backbone} + linear-attention decoder, "
                    f"{_DEFAULT_ITERS} iters"
                ),
                tags=("trained", "multiclass-capable", "reconstruction", "sota"),
                trainable=True,
                allowed_regimes=("multiclass",),
            )
        )


# -----------------------------------------------------------------------------
# Anomalib adapter: the ~25 gradient-trained methods this project does not
# reimplement (EfficientAD, GLASS, Reverse Distillation, the normalizing-flow
# family, ...)
# -----------------------------------------------------------------------------
#
# Reimplementing a dozen papers here would be a dozen chances to get one
# subtly wrong and publish a misleading number. So those are delegated to
# Anomalib, which already maintains tested implementations, and this section
# makes them look identical to a native method from the caller's point of
# view: fit on normal images, return per-image scores. Metrics are then
# computed by this project's own evaluation code from raw scores, not read
# out of Anomalib's own metric objects, so both halves of the zoo are
# measured the same way.
#
# Anomalib is an optional dependency (see the ``try/except ImportError`` at
# the top of this module). If it is not installed, no anomalib_* methods are
# registered and the native zoo above still runs:
#
#     pip install anomalib
#
# Two honest caveats about anything reported from this backend:
#
# * Anomalib models use their own default input geometry, so they do not see
#   the per-dataset aspect ratios in ``training.data.DEFAULT_SIZE``. Their
#   numbers are comparable to each other and broadly comparable to the native
#   methods, but a small gap on the non-square datasets (Kolektor, Severstal)
#   may be geometry rather than method.
# * They fit by gradient descent, so unlike the native half their results
#   depend on epoch count and seed.

#: Zoo entries as ``benchmark name -> (Anomalib class name, default epochs)``.
#: Epoch counts follow each paper's own recipe where it has one; the
#: training-free entries (Patchcore, Padim, Dfm, Dfkde) use a single pass.
#: Names are resolved with ``getattr`` at import time, so an entry missing from
#: the installed Anomalib version is skipped rather than crashing the sweep.
_ANOMALIB_ZOO: dict[str, tuple[str, int]] = {
    # Memory-bank / statistical -- Anomalib's own take on the native methods,
    # worth running as a cross-check that our implementations are faithful.
    "patchcore": ("Patchcore", 1),
    "padim": ("Padim", 1),
    "dfm": ("Dfm", 1),
    "dfkde": ("Dfkde", 1),
    # Student-teacher / distillation.
    "stfpm": ("Stfpm", 100),
    "reverse_distillation": ("ReverseDistillation", 200),
    "efficient_ad": ("EfficientAd", 100),
    "fre": ("Fre", 100),
    # Normalizing flows.
    "fastflow": ("Fastflow", 200),
    "cflow": ("Cflow", 50),
    "csflow": ("Csflow", 100),
    "uflow": ("Uflow", 200),
    # Reconstruction / synthesis.
    "draem": ("Draem", 100),
    "dsr": ("Dsr", 100),
    "ganomaly": ("Ganomaly", 100),
    "glass": ("Glass", 100),
    "supersimplenet": ("SuperSimpleNet", 100),
    # Foundation-model based -- the current top of the leaderboard.
    "dinomaly": ("Dinomaly", 100),
    "uninet": ("UniNet", 100),
    "inpformer": ("INPFormer", 100),
    "generalad": ("GeneralAD", 100),
    "anomaly_dino": ("AnomalyDINO", 1),
    "anomalyvfm": ("AnomalyVFM", 1),
    "superadd": ("SuperADD", 1),
    # Zero-/few-shot vision-language.
    "winclip": ("WinClip", 1),
    # Feature adaptation.
    "cfa": ("Cfa", 30),
}


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialize ``source`` at ``destination``, preferring a hard link.

    Anomalib addresses data by directory, not by path list, so the split has
    to exist on disk. Hard links make that free: no second copy of MVTec, and
    it works on NTFS without the elevation that symlinks need.
    """
    try:
        destination.hardlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def _materialize_split(
    config: DatasetConfig,
    train_paths: Sequence[Path],
    test_split: Sequence[tuple[Path, bool]],
    root: Path,
) -> Path:
    """Lay out one config as the ``normal``/``abnormal``/``normal_test`` tree.

    Filenames are prefixed with their index because the source corpora reuse
    names across sub-folders (every MVTec defect class restarts at ``000.png``)
    and a flat destination directory would silently drop the collisions.

    Args:
        config: The configuration being materialized.
        train_paths: Normal-only training images.
        test_split: ``(path, is_anomalous)`` test pairs.
        root: Directory to build the tree under.

    Returns:
        The dataset root containing the three sub-directories.
    """
    dataset_root = root / config.slug
    for name in ("normal", "abnormal", "normal_test"):
        (dataset_root / name).mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(train_paths):
        _link_or_copy(path, dataset_root / "normal" / f"{index:06d}{path.suffix}")

    normal_index = anomalous_index = 0
    for path, is_anomalous in test_split:
        if is_anomalous:
            target = dataset_root / "abnormal" / f"{anomalous_index:06d}{path.suffix}"
            anomalous_index += 1
        else:
            target = dataset_root / "normal_test" / f"{normal_index:06d}{path.suffix}"
            normal_index += 1
        _link_or_copy(path, target)

    return dataset_root


class AnomalibScorer:
    """A fitted Anomalib model, replayed to produce per-image scores.

    Anomalib owns the fit/predict loop, so unlike the native scorers this one
    cannot score an arbitrary path list after the fact: the scores for the
    materialized test split are computed once during the fit. It also chooses
    its own iteration order, so it carries the matching ground-truth labels in
    :attr:`labels` and the caller scores against those rather than its own.

    Args:
        scores: Per-image anomaly scores in Anomalib's prediction order.
        labels: Ground-truth anomaly labels in that same order.
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Store the precomputed scores and their matching labels."""
        self._scores = scores
        self.labels = labels

    def score(self, paths: list[Path]) -> np.ndarray:
        """Return the precomputed scores.

        Raises:
            RuntimeError: If asked for a different number of images than were
                scored during the fit, which would mean the caller and the
                adapter disagree about the split.
        """
        if len(paths) != len(self._scores):
            msg = (
                f"Anomalib scorer holds {len(self._scores)} scores but was asked "
                f"for {len(paths)}; the test split changed between fit and score."
            )
            raise RuntimeError(msg)
        return self._scores


def _build_datamodule(dataset_root: Path, config: DatasetConfig, options: RunOptions) -> Folder:
    """Construct a Folder datamodule over a materialized split."""
    return Folder(
        name=config.slug,
        root=dataset_root,
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_test_dir="normal_test",
        train_batch_size=options.batch_size,
        eval_batch_size=options.batch_size,
        num_workers=options.num_workers,
    )


def _collect_scores(predictions: object) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Anomalib prediction batches into score and label arrays.

    Attribute names have moved around across Anomalib versions, so this reads
    defensively rather than assuming one shape.

    Returns:
        ``(scores, labels)`` as 1-D float and bool arrays.
    """
    scores: list[float] = []
    labels: list[bool] = []
    for batch in predictions or []:  # type: ignore[union-attr]
        batch_scores = getattr(batch, "pred_score", None)
        batch_labels = getattr(batch, "gt_label", None)
        if batch_scores is None:
            msg = "Anomalib prediction batch has no 'pred_score'; unsupported version."
            raise RuntimeError(msg)
        scores.extend(np.atleast_1d(np.asarray(batch_scores.detach().cpu())).ravel().tolist())
        if batch_labels is not None:
            labels.extend(
                np.atleast_1d(np.asarray(batch_labels.detach().cpu())).ravel().astype(bool).tolist()
            )
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


def _fit_anomalib(class_name: str, default_epochs: int):
    """Build a fit function that trains one Anomalib model and caches its scores."""

    def fit(
        config: DatasetConfig,
        paths: list[Path],
        test_split: list[tuple[Path, bool]],
        options: RunOptions,
    ) -> AnomalibScorer:
        model_class = getattr(anomalib_models, class_name)
        epochs = options.epochs or default_epochs

        workdir = Path(tempfile.mkdtemp(prefix=f"anomalib_{config.slug}_"))
        try:
            dataset_root = _materialize_split(config, paths, test_split, workdir / "data")
            datamodule = _build_datamodule(dataset_root, config, options)
            model = model_class()
            engine = Engine(
                max_epochs=epochs,
                accelerator="gpu" if options.device.startswith("cuda") else "cpu",
                devices=1,
                default_root_dir=str(workdir / "runs"),
                logger=False,
            )
            engine.fit(model=model, datamodule=datamodule)
            predictions = engine.predict(model=model, datamodule=datamodule)
            scores, labels = _collect_scores(predictions)

            if labels.size != scores.size:
                msg = (
                    f"Anomalib returned {scores.size} scores but {labels.size} labels; "
                    "cannot align predictions to ground truth."
                )
                raise RuntimeError(msg)
            return AnomalibScorer(scores, labels)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return fit


def _register_anomalib_methods() -> None:
    """Register every zoo entry present in the installed Anomalib version."""
    if not ANOMALIB_AVAILABLE:
        return
    for name, (class_name, epochs) in _ANOMALIB_ZOO.items():
        if not hasattr(anomalib_models, class_name):
            continue
        register(
            MethodSpec(
                name=f"anomalib_{name}",
                family="anomalib",
                backend="anomalib",
                fit=_fit_anomalib(class_name, epochs),
                exportable=False,
                notes=f"Anomalib {class_name}, {epochs} epoch(s)",
                tags=("anomalib", "trained" if epochs > 1 else "training-free"),
            )
        )


_register_native_methods()
_register_dinomaly_methods()
_register_anomalib_methods()


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def get_model(name: str, backbone: str | None = None) -> MethodSpec:
    """One factory for the whole zoo: look up a method by name, optionally split by backbone.

    Two calling conventions:

    * ``get_model("patchcore_dinov2_vitb14")`` -- the exact registered name.
    * ``get_model("patchcore", "dinov2_vitb14")`` -- family/method plus
      backbone, joined as ``f"{name}_{backbone}"`` before lookup. This is how
      method names in this zoo are actually built (see
      :func:`_register_native_methods`), so it works for every native
      method; Dinomaly and Anomalib entries (which have no separate backbone
      argument) should be looked up by exact name instead.

    Args:
        name: A registered method name, or a method/family prefix to combine
            with ``backbone``.
        backbone: Optional backbone key to append to ``name``.

    Returns:
        The matching :class:`MethodSpec`, including its ``fit`` callable.

    Raises:
        SystemExit: If the resolved name is not registered. Use
            :func:`all_methods` to list what's available.
    """
    full_name = f"{name}_{backbone}" if backbone else name
    return get(full_name)
