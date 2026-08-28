"""EncoderDecoder: the standard backbone + decode_head segmentor.

Mirrors MMSeg's EncoderDecoder in structure:
    backbone → decode_head → logits

Built entirely from config dicts, e.g.:

    model = EncoderDecoder(
        backbone=dict(type='ResNet', depth=50, pretrained=True),
        decode_head=dict(type='FCNHead', in_channels=2048, num_classes=19),
    )
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from semseg.utils.registry import BACKBONES, DECODE_HEADS, SEGMENTORS

from .base_segmentor import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Standard encoder-decoder segmentor.

    Args:
        backbone:     Config dict for the backbone (must have 'type').
        decode_head:  Config dict for the decode head (must have 'type').
        auxiliary_head: Optional auxiliary decode head config.
                        Used during training only (e.g. FCNHead on layer3).
        aux_loss_weight: Weight applied to the auxiliary loss (default: 0.4).
        train_cfg:    Placeholder — reserved for future training options.
        test_cfg:     Placeholder — reserved for future test-time options.

    Example::

        from semseg.models.segmentors import EncoderDecoder

        model = EncoderDecoder(
            backbone=dict(
                type='ResNet',
                depth=50,
                output_stride=16,
                pretrained=True,
            ),
            decode_head=dict(
                type='FCNHead',
                in_channels=2048,
                in_index=-1,
                channels=256,
                num_classes=19,
                num_convs=2,
                dropout_ratio=0.1,
            ),
        )
        img = torch.randn(2, 3, 512, 512)
        logits = model(img)  # inference → (2, 19, 512, 512)
    """

    def __init__(
        self,
        backbone: Dict,
        decode_head: Dict,
        auxiliary_head: Optional[Dict] = None,
        aux_loss_weight: float = 0.4,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
    ):
        super().__init__()

        self.backbone = BACKBONES.build(backbone)
        self.decode_head = DECODE_HEADS.build(decode_head)

        self.with_auxiliary_head = auxiliary_head is not None
        if self.with_auxiliary_head:
            self.auxiliary_head = DECODE_HEADS.build(auxiliary_head)
        self.aux_loss_weight = aux_loss_weight

        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

    # ── BaseSegmentor interface ───────────────────────────────────────────────

    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(img)

    def encode_decode(
        self,
        img: torch.Tensor,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        feats = self.extract_feat(img)
        return self.decode_head(feats, img_size=img_size)

    def forward_train(
        self,
        img: torch.Tensor,
        gt_semantic_seg: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            img:             (B, 3, H, W) — normalised input image.
            gt_semantic_seg: (B, H, W)   — integer class labels, ignore=-1 or 255.

        Returns:
            Dict with key 'loss_seg' and optionally 'loss_aux'.
            Caller is responsible for summing and calling .backward().
        """
        feats = self.extract_feat(img)
        img_size = (img.shape[2], img.shape[3])

        # Main head loss
        logits = self.decode_head(feats, img_size=img_size)
        losses = self._seg_loss(logits, gt_semantic_seg, name="loss_seg")

        # Auxiliary head loss (uses an earlier feature, e.g. layer3)
        if self.with_auxiliary_head:
            aux_logits = self.auxiliary_head(feats, img_size=img_size)
            aux_losses = self._seg_loss(aux_logits, gt_semantic_seg, name="loss_aux")
            # Scale and merge
            for k, v in aux_losses.items():
                losses[k] = v * self.aux_loss_weight

        return losses

    def forward_test(self, img: torch.Tensor) -> torch.Tensor:
        img_size = (img.shape[2], img.shape[3])
        return self.encode_decode(img, img_size=img_size)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _seg_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = 255,
        name: str = "loss_seg",
    ) -> Dict[str, torch.Tensor]:
        """Cross-entropy loss with ignore_index.

        NOTE: This is a built-in fallback.  When the loss registry is added,
        this will be replaced by LOSSES.build(cfg).

        Args:
            logits:  (B, C, H, W) — raw un-normalised logits.
            targets: (B, H, W)   — integer class labels.
            ignore_index: Label value to ignore (default 255, Cityscapes-style).
            name:    Key for the returned dict.

        Returns:
            {name: scalar tensor}
        """
        loss = F.cross_entropy(logits, targets.long(), ignore_index=ignore_index)
        return {name: loss}
