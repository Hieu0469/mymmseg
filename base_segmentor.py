"""Abstract base class for all segmentors (full models)."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseSegmentor(nn.Module, ABC):
    """Abstract base class for segmentation models.

    A segmentor wires together:
        backbone → (optional neck) → decode_head → logits

    Subclasses must implement:
        - forward_train()   training pass returning a loss dict
        - forward_test()    inference pass returning logit or pred map
        - encode_decode()   shared backbone + decode step

    The base class provides:
        - Unified forward() dispatching train/test mode
        - extract_feat()    runs the backbone
    """

    def __init__(self):
        super().__init__()

    # ── Must-implement API ────────────────────────────────────────────────────

    @abstractmethod
    def encode_decode(
        self,
        img: torch.Tensor,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Run backbone + decode head.

        Args:
            img:      (B, 3, H, W)
            img_size: Target size for upsample; None = no upsample.

        Returns:
            Logit map (B, num_classes, h, w).
        """
        ...

    @abstractmethod
    def forward_train(
        self,
        img: torch.Tensor,
        gt_semantic_seg: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward.

        Returns:
            Dict of named losses, e.g. {'loss_seg': tensor}.
        """
        ...

    @abstractmethod
    def forward_test(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """Inference forward.

        Returns:
            Logit map (B, num_classes, H, W) — full image resolution.
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Run the backbone only. Subclasses override if a neck is present."""
        return self.backbone(img)

    # ── Unified forward ───────────────────────────────────────────────────────

    def forward(
        self,
        img: torch.Tensor,
        gt_semantic_seg: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Training:   forward(img, gt_semantic_seg) → loss dict
        Inference:  forward(img)                  → logit map
        """
        if self.training:
            if gt_semantic_seg is None:
                raise ValueError("gt_semantic_seg is required during training.")
            return self.forward_train(img, gt_semantic_seg)
        else:
            return self.forward_test(img)

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        """Convenience: argmax of forward_test output → class map (B, H, W)."""
        logits = self.forward_test(img)
        return logits.argmax(dim=1)
