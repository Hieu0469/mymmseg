"""Abstract base class for all backbones."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract base class that all backbones must inherit from.

    A backbone receives an image tensor and returns a list of feature maps
    at multiple scales (multi-scale feature hierarchy), e.g.:
        [C2, C3, C4, C5]  — stride 4, 8, 16, 32

    Subclasses must implement:
        - forward(x) -> List[Tensor]
        - out_channels (property) -> List[int]

    Optionally override:
        - init_weights()    for pretrained loading
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        """Number of channels for each output feature map, low → high stride."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            List of feature tensors [f1, f2, ...] from low to high stride.
            Each fi has shape (B, C_i, H/s_i, W/s_i).
        """
        ...

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Load pretrained weights. Override in subclasses if needed."""
        pass
