"""ResNet backbone with dilated convolutions support (DeepLab-style).

Wraps torchvision's ResNet and exposes multi-scale feature outputs.
Supports output_stride=8 or 16 via dilation (no stride in layer3/layer4).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)

from ...utils.registry import BACKBONES

from .base_backbone import BaseBackbone


# ── Helpers ───────────────────────────────────────────────────────────────────

_RESNET_BUILDERS = {
    18: (resnet18, ResNet18_Weights.DEFAULT),
    34: (resnet34, ResNet34_Weights.DEFAULT),
    50: (resnet50, ResNet50_Weights.DEFAULT),
    101: (resnet101, ResNet101_Weights.DEFAULT),
}

# Channels per stage: [layer1, layer2, layer3, layer4]
_RESNET_CHANNELS = {
    18:  [64,  128, 256,  512],
    34:  [64,  128, 256,  512],
    50:  [256, 512, 1024, 2048],
    101: [256, 512, 1024, 2048],
}


def _make_layer_dilated(layer: nn.Module, dilation: int) -> None:
    """Patch stride→1 and add dilation to every conv in a ResNet layer in-place."""
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)


# ── Backbone ──────────────────────────────────────────────────────────────────

@BACKBONES.register_module()
class ResNet(BaseBackbone):
    """ResNet backbone with optional dilated output stride.

    Args:
        depth: ResNet depth — one of {18, 34, 50, 101}.
        output_stride: Effective output stride of the backbone.
            - 32 (default): standard ResNet, no dilation.
            - 16: layer4 uses dilation=2 instead of stride=2.
            - 8:  layer3 + layer4 both dilated.
        out_indices: Which stages to return. 0→layer1, 1→layer2, ...
            Defaults to (0, 1, 2, 3) — all four stages.
        frozen_stages: Freeze stem + first N stages. -1 = nothing frozen.
        pretrained: If True, load ImageNet weights via torchvision.
    """

    def __init__(
        self,
        depth: int = 50,
        output_stride: int = 32,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        pretrained: bool = False,
    ):
        super().__init__()

        if depth not in _RESNET_BUILDERS:
            raise ValueError(f"depth must be one of {list(_RESNET_BUILDERS)}, got {depth}")
        if output_stride not in (8, 16, 32):
            raise ValueError(f"output_stride must be 8, 16, or 32, got {output_stride}")

        self.depth = depth
        self.output_stride = output_stride
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # ── Build torchvision ResNet ───────────────────────────────────────
        builder, weights = _RESNET_BUILDERS[depth]
        tv_model = builder(weights=weights if pretrained else None)

        # ── Extract sub-modules (MMSeg naming convention) ──────────────────
        self.conv1 = tv_model.conv1    # stride 2
        self.bn1   = tv_model.bn1
        self.relu  = tv_model.relu
        self.maxpool = tv_model.maxpool  # stride 2  → total: stride 4

        self.layer1 = tv_model.layer1  # stride 4
        self.layer2 = tv_model.layer2  # stride 8
        self.layer3 = tv_model.layer3  # stride 16
        self.layer4 = tv_model.layer4  # stride 32

        # ── Apply dilation if output_stride < 32 ─────────────────────────
        if output_stride == 16:
            _make_layer_dilated(self.layer4, dilation=2)
        elif output_stride == 8:
            _make_layer_dilated(self.layer3, dilation=2)
            _make_layer_dilated(self.layer4, dilation=4)

        self._channels = _RESNET_CHANNELS[depth]
        self._freeze_stages()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def out_channels(self) -> List[int]:
        """Channels for each requested output index."""
        return [self._channels[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            List of feature tensors for each index in self.out_indices.
            Strides (output_stride=32): [4, 8, 16, 32]
            Strides (output_stride=16): [4, 8, 16, 16]
            Strides (output_stride=8):  [4, 8,  8,  8]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        outs: List[torch.Tensor] = []
        for i, stage in enumerate(stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)

        return outs

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _freeze_stages(self) -> None:
        if self.frozen_stages < 0:
            return

        # Freeze stem
        for m in [self.conv1, self.bn1]:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        # Freeze layer1 … layerN
        for i in range(self.frozen_stages):
            stage = getattr(self, f"layer{i + 1}")
            stage.eval()
            for p in stage.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """Keep frozen stages in eval mode even when the model is set to train."""
        super().train(mode)
        self._freeze_stages()
        return self