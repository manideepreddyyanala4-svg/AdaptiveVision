"""Training-free embedding methods, implemented directly on the shared trunk.

These are the methods that need no gradient descent at all: fitting is one
forward pass over the normal images plus some linear algebra. That matters for
a production line -- there is no learning rate to tune, no epoch count to
babysit, and re-fitting after a process change is minutes rather than hours --
and it keeps every one of them exportable as a single ONNX graph.

Three scoring rules are implemented, each over any backbone in
``benchmark.backbones``:

* **PaDiM** -- a Gaussian per patch position, scored by Mahalanobis distance.
  Strong when the part is camera-aligned, since position ``(i, j)`` then means
  the same thing in every image.
* **PatchCore** -- a coreset-subsampled memory bank of normal patches, scored
  by nearest-neighbour distance. Position-agnostic, so it handles unaligned
  material (steel strip) that PaDiM's per-position prior gets wrong.
* **DFM** -- a PCA subspace over globally pooled features, scored by
  reconstruction error. Much cheaper than either, and a useful control: when
  it wins, the defect signal was global and the patch machinery bought nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark.backbones import BACKBONES, build_extractor
from benchmark.data import DatasetConfig
from benchmark.loading import image_loader
from benchmark.registry import MethodSpec, RunOptions, register

#: Patch features are projected to this many dimensions before they are
#: stored or compared. Full WideResNet/DINOv2 features are 1024-2048 channels
#: wide; the papers all reduce, and on an 8 GB card it is the difference
#: between fitting and not.
_PADIM_DIMS = 100
_PATCHCORE_DIMS = 384

#: Upper bound on patches held in memory while building a PatchCore bank.
#: Severstal alone would otherwise pool ~1.6M patches.
_MAX_POOL_PATCHES = 250_000


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
    ``benchmark.export`` trace the whole scorer -- backbone included -- into
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
                if estimated > _MAX_POOL_PATCHES:
                    per_image_budget = max(1, _MAX_POOL_PATCHES // max(1, len(paths)))
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
        indices = _greedy_coreset(pooled.to(self.device).float(), target, self.options.seed)
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


def _greedy_coreset(features: torch.Tensor, target: int, seed: int) -> torch.Tensor:
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


def _register_all() -> None:
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


_register_all()
