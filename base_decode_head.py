"""Abstract base class for all decode heads."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecodeHead(nn.Module, ABC):
    """Abstract base class for segmentation decode heads.

    A decode head takes the multi-scale backbone features and produces
    a logit map of shape (B, num_classes, H, W).

    Subclasses must implement:
        - _build_head()        called in __init__ to build layers
        - forward_features()   takes selected features → raw logit map (before upsample)

    The base class handles:
        - Feature selection via `in_index`
        - Final upsampling to input resolution
        - align_corners flag

    Args:
        in_channels:   Channel count(s) for the selected backbone features.
                       int for single feature, list for multiple.
        in_index:      Which backbone output(s) to use. -1 = last feature.
        channels:      Internal working channels of the head.
        num_classes:   Number of segmentation classes.
        dropout_ratio: Dropout before the final conv. 0 = disabled.
        align_corners: Passed to F.interpolate.
    """

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        in_index: Union[int, List[int]] = -1,
        channels: int = 256,
        num_classes: int = 19,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        # Dropout before cls_seg (shared by most heads)
        self.dropout = (
            nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )

        # Final classification conv — always (channels → num_classes, k=1)
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        # Let subclasses build their head-specific layers
        self._build_head()

    # ── Must-implement API ────────────────────────────────────────────────────

    @abstractmethod
    def _build_head(self) -> None:
        """Build all layers specific to this decode head."""
        ...

    @abstractmethod
    def forward_features(
        self, inputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Core decode logic.

        Args:
            inputs: Selected feature tensor(s) from _select_features().

        Returns:
            Logit map (B, channels, h, w) — NOT yet upsampled or classified.
            The base class will apply dropout → cls_seg → upsample.
        """
        ...

    # ── Base class forward ────────────────────────────────────────────────────

    def forward(
        self,
        inputs: List[torch.Tensor],
        img_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs:   List of all backbone feature tensors.
            img_size: Target (H, W) to upsample the logit map to.
                      If None, no upsampling is applied (useful for loss on
                      downsampled targets).

        Returns:
            Logit map (B, num_classes, H, W).
        """
        x = self._select_features(inputs)
        x = self.forward_features(x)       # subclass decode
        x = self.dropout(x)
        x = self.cls_seg(x)                # → (B, num_classes, h, w)

        if img_size is not None:
            x = F.interpolate(
                x,
                size=img_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return x

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _select_features(
        self, inputs: List[torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Select backbone feature(s) according to self.in_index."""
        if isinstance(self.in_index, int):
            return inputs[self.in_index]
        else:
            return [inputs[i] for i in self.in_index]
