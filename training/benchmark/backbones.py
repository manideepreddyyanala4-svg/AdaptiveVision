"""Frozen pretrained feature extractors shared by every native method.

The single biggest lever on embedding-based anomaly detection is not the
scoring rule but the features it scores. ``padim.py`` hardcoded two ResNet
trunks; this module turns the backbone into a swappable axis of the sweep, so
PaDiM-on-ResNet18 and PatchCore-on-DINOv2 become the same experiment with two
knobs changed rather than two separate codebases.

Everything here is plain tensor ops on a frozen trunk -- no gradients, no
training -- so a scorer fitted on top of one of these extractors still exports
to a single ONNX graph matching the production contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

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
