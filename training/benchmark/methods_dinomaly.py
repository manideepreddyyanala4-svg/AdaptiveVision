"""Training and registration for the Dinomaly family.

Everything else in the native zoo fits in a single forward pass. Dinomaly is
the one method here that needs gradient descent, so it owns its own training
loop rather than pretending to fit the training-free mould -- but it exposes
the same ``score`` / ``score_with_maps`` surface, so the runner, the metrics,
the ensembler and the dashboard treat it identically.

The recipe follows the reference implementation: 448px resized to a 392px
crop, StableAdamW-style AdamW at 2e-3 with amsgrad, warm-cosine decay to 2e-4,
and a hard-mining percentile ramped to 0.9 over the first 1000 iterations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark.data import DatasetConfig
from benchmark.dinomaly import (
    DINOMALY_BACKBONES,
    DinomalyModel,
    WarmCosineSchedule,
    global_cosine_hard_mining,
)
from benchmark.loading import image_loader
from benchmark.registry import MethodSpec, RunOptions, register

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
#: ``benchmark.data`` are deliberately overridden here.
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
        # below) -- exposed so a caller timing inference (cost.py) uses the
        # real input shape instead of the dataset's, which would otherwise
        # either crash (not a multiple of 14) or silently time the wrong size.
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
        backbone: Key into :data:`~benchmark.dinomaly.DINOMALY_BACKBONES`.
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


def _register_all() -> None:
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


_register_all()
