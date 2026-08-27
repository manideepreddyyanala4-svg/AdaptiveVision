"""Autoencoder architecture and the ONNX-export wrapper.

``ConvAutoencoder`` is trained on normal-only images with an MSE
reconstruction loss. ``AnomalyExportModel`` wraps the trained autoencoder so
the *exported ONNX graph itself* matches the production contract consumed by
``adaptivevision.inspection.anomaly.detector.ThresholdAnomalyDetector``:
input named ``"input"`` of static shape ``(1, H, W)`` (channel dim only, no
batch axis), output named ``"output"`` -- a single scalar in ``[0, 1]``.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for single-channel square images.

    ``image_size`` must be divisible by 16 (four stride-2 downsamples).
    """

    def __init__(self, image_size: int, latent_channels: int = 128) -> None:
        super().__init__()
        if image_size % 16 != 0:
            msg = f"image_size must be divisible by 16, got {image_size}"
            raise ValueError(msg)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x`` (shape ``(N, 1, H, W)``, values in ``[0, 1]``)."""
        return self.decoder(self.encoder(x))


class AnomalyExportModel(nn.Module):
    """Wraps a trained autoencoder into the exact production ONNX contract.

    Args:
        autoencoder: Trained, frozen ``ConvAutoencoder``.
        mu: Mean reconstruction score on held-out normal validation data.
        sigma: Standard deviation of that score (floor-clamped by the caller).
        topk_fraction: Fraction of worst-reconstructed pixels averaged into
            the raw anomaly score, so localized defects aren't diluted by a
            plain whole-image MSE.
    """

    def __init__(
        self,
        autoencoder: ConvAutoencoder,
        mu: float,
        sigma: float,
        topk_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.topk_fraction = topk_fraction

    def raw_score(self, x_chw_0_1: torch.Tensor) -> torch.Tensor:
        """Top-k-fraction mean squared reconstruction error for a batch."""
        recon = self.autoencoder(x_chw_0_1)
        err = (recon - x_chw_0_1) ** 2
        err_flat = err.reshape(err.shape[0], -1)
        k = max(1, int(err_flat.shape[1] * self.topk_fraction))
        topk = torch.topk(err_flat, k, dim=1).values
        return topk.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a single frame ``x`` of shape ``(1, H, W)``, values ``[0, 255]``.

        Returns:
            A ``(1,)`` tensor with a calibrated anomaly score in ``[0, 1]``.
        """
        batched = (x / 255.0).unsqueeze(0)
        raw = self.raw_score(batched)
        score = torch.sigmoid((raw - self.mu) / self.sigma)
        return score.reshape(1)
