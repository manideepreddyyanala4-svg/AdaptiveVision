"""PaDiM: pretrained-backbone patch distribution modeling for anomaly detection.

Reconstruction autoencoders trained from scratch on a few hundred images (see
``model.py``/``train_anomaly_model.py``) top out well below what's needed for
a strong result. PaDiM (Defard et al., 2020) instead extracts multi-layer
features from an ImageNet-pretrained CNN (which already "knows" what normal
textures/edges look like), fits a per-patch-position Gaussian from the
normal-only training images, and scores new images by Mahalanobis distance to
those Gaussians. It needs no gradient descent at all -- fitting is a single
forward pass over the training images plus a covariance/inverse computation.

Everything here (backbone forward, channel selection, Mahalanobis distance,
sigmoid calibration) is plain tensor ops, so the *whole* scorer -- backbone
included -- exports to one ONNX graph matching the same production contract
used by the autoencoder path: input ``"input"`` of static shape ``(3, H, W)``,
output ``"output"`` -- a scalar anomaly score in ``[0, 1]``.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)

#: Per-backbone (layer1, layer2, layer3) channel counts, so the caller can
#: size the concatenated feature map without instantiating the backbone.
_BACKBONE_CHANNELS = {
    "resnet18": (64, 128, 256),
    "wide_resnet50_2": (256, 512, 1024),
}


class PaDiMFeatureExtractor(nn.Module):
    """Frozen backbone trunk producing a concatenated multi-layer feature map.

    ``wide_resnet50_2`` matches PaDiM's best-reported results but costs more
    compute/memory than ``resnet18``; both are ImageNet-pretrained and frozen.
    """

    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()
        if backbone_name == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == "wide_resnet50_2":
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        else:
            msg = f"Unknown backbone: {backbone_name!r}"
            raise ValueError(msg)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self._num_channels = sum(_BACKBONE_CHANNELS[backbone_name])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def num_channels(self) -> int:
        """Total channel count of the concatenated feature map."""
        return self._num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch ``x`` of shape ``(N, 3, H, W)``, values ``[0, 255]``."""
        x = (x / 255.0 - self.mean) / self.std
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        size = f1.shape[-2:]
        f2_up = nn.functional.interpolate(f2, size=size, mode="nearest")
        f3_up = nn.functional.interpolate(f3, size=size, mode="nearest")
        return torch.cat([f1, f2_up, f3_up], dim=1)


class PaDiMModel(nn.Module):
    """Wraps the extractor + fitted Gaussians into the production ONNX contract.

    Args:
        extractor: Feature extractor (frozen).
        channel_indices: Fixed random subset of feature channels, shape ``(d,)``.
        mean: Per-patch-position mean, shape ``(P, d)``.
        inv_cov: Per-patch-position inverse covariance, shape ``(P, d, d)``.
        mu: Mean raw Mahalanobis score on held-out normal validation data.
        sigma: Standard deviation of that score (floor-clamped by the caller).
    """

    def __init__(
        self,
        extractor: PaDiMFeatureExtractor,
        channel_indices: torch.Tensor,
        mean: torch.Tensor,
        inv_cov: torch.Tensor,
        mu: float,
        sigma: float,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.register_buffer("channel_indices", channel_indices)
        self.register_buffer("mean", mean)
        self.register_buffer("inv_cov", inv_cov)
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reduced-dimension per-patch feature matrix ``(P, d)`` for one image."""
        features = self.extractor(x.unsqueeze(0))
        selected = torch.index_select(features, 1, self.channel_indices)
        _, channels, height, width = selected.shape
        return selected.reshape(channels, height * width).transpose(0, 1)

    def raw_score(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """Max per-patch Mahalanobis distance over all patch positions."""
        diff = patch_feats - self.mean
        weighted = torch.bmm(diff.unsqueeze(1), self.inv_cov).squeeze(1)
        mahalanobis = (weighted * diff).sum(dim=1)
        return mahalanobis.max()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a single frame ``x`` of shape ``(3, H, W)``, values ``[0, 255]``.

        Returns:
            A ``(1,)`` tensor with a calibrated anomaly score in ``[0, 1]``.
        """
        raw = self.raw_score(self.patch_features(x))
        score = torch.sigmoid((raw - self.mu) / self.sigma)
        return score.reshape(1)


def fit_gaussians(
    extractor: PaDiMFeatureExtractor,
    features_nd: torch.Tensor,
    eps: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a per-patch-position Gaussian from stacked training features.

    Args:
        extractor: Unused directly; kept for signature symmetry/clarity.
        features_nd: Stacked reduced-dimension features, shape ``(N, P, d)``.
        eps: Diagonal regularization added before inversion.

    Returns:
        ``(mean, inv_cov)`` of shapes ``(P, d)`` and ``(P, d, d)``.
    """
    _ = extractor
    n = features_nd.shape[0]
    mean = features_nd.mean(dim=0)
    centered = (features_nd - mean).permute(1, 0, 2).contiguous()  # (P, N, d)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (n - 1)  # (P, d, d)
    d = cov.shape[-1]
    identity = torch.eye(d, device=cov.device).unsqueeze(0)
    cov = cov + eps * identity
    inv_cov = torch.linalg.inv(cov)
    return mean, inv_cov


def fit_gaussian_pooled(
    features_nd: torch.Tensor,
    eps: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a single position-agnostic Gaussian, pooling all patches together.

    Per-position modeling (:func:`fit_gaussians`) assumes patch position
    ``(i, j)`` means the same thing across training images -- true for
    centered, camera-aligned parts (MVTec/VisA/Kolektor), but false for
    arbitrary crops of continuous material (e.g. Severstal steel strips),
    where a defect can appear at any position and unrelated positions across
    images share no semantic alignment. Pooling every patch from every
    position into one Gaussian is the position-agnostic analogue.

    Returns:
        ``(mean, inv_cov)`` broadcast to per-position shapes ``(P, d)`` and
        ``(P, d, d)`` (repeating the single Gaussian ``P`` times) so the
        result is a drop-in replacement for :func:`fit_gaussians` -- same
        contract, same downstream ``PaDiMModel``.
    """
    n, p, d = features_nd.shape
    pooled = features_nd.reshape(n * p, d)
    mean = pooled.mean(dim=0)
    centered = pooled - mean
    cov = (centered.T @ centered) / (n * p - 1)
    cov = cov + eps * torch.eye(d, device=cov.device)
    inv_cov = torch.linalg.inv(cov)
    mean_p = mean.unsqueeze(0).expand(p, -1).contiguous()
    inv_cov_p = inv_cov.unsqueeze(0).expand(p, -1, -1).contiguous()
    return mean_p, inv_cov_p
