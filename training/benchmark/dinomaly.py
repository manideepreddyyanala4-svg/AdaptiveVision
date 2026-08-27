"""Dinomaly: the multi-class reconstruction model this benchmark is built around.

Every other method here fits one model per category. That is the setting the
literature reports, and it is a poor fit for a real line: 29 categories means
29 checkpoints to version, deploy, calibrate and monitor, and a new product
means a new model. Dinomaly (Guo et al., CVPR 2025) is the first method to
close that gap -- one checkpoint covering every category at accuracy that
matches per-category specialists.

The design is deliberately plain: a frozen DINOv2 encoder, an MLP bottleneck,
and a small Transformer decoder trained to reconstruct the encoder's features
for normal images. Anomalies are wherever the reconstruction fails. The paper's
contribution is four restraints that stop the decoder learning to reconstruct
*everything*, which would leave nothing for the anomaly score to detect:

* **Frozen foundation encoder.** DINOv2 features already separate normal from
  abnormal; fine-tuning them destroys that.
* **Noisy bottleneck.** Dropout at 0.2 on the bottleneck MLP injects noise, so
  the decoder cannot memorize an identity mapping.
* **Linear attention.** Its inability to focus sharply is the point -- a
  softmax decoder attends precisely enough to copy anomalies through.
* **Loose reconstruction.** Layers are fused into two groups rather than
  matched one-to-one, and a hard-mining cosine loss down-weights the easy
  points that would otherwise dominate the gradient.

Implemented natively rather than pulled from Anomalib so it can be trained in
the multi-class regime, exported, and cross-checked against Anomalib's own
implementation -- two independent implementations agreeing is worth more than
one number.
"""

from __future__ import annotations

import math

import timm
import torch
from torch import nn

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
