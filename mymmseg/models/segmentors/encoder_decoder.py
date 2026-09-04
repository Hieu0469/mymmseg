"""EncoderDecoder segmentor — mirrors mmseg's EncoderDecoder.

Built from registry config dicts:

    model = EncoderDecoder(
        backbone=dict(type='ResNet', depth=50, ...),
        decode_head=dict(type='FCNHead', in_channels=2048, num_classes=19),
    )
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...utils.registry import BACKBONES, DECODE_HEADS, SEGMENTORS
from .base_segmentor import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Standard backbone + decode_head segmentor.

    Args:
        backbone: Config dict for the backbone.
        decode_head: Config dict for the main decode head.
        auxiliary_head: Optional auxiliary head config (training only).
        aux_loss_weight: Weight for auxiliary loss. Default: 0.4.
        train_cfg / test_cfg: Reserved for future use.
    """

    def __init__(
        self,
        backbone: Dict,
        decode_head: Dict,
        auxiliary_head: Optional[Dict] = None,
        aux_loss_weight: float = 0.4,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.decode_head = DECODE_HEADS.build(decode_head)

        self.with_auxiliary_head = auxiliary_head is not None
        if self.with_auxiliary_head:
            self.auxiliary_head = DECODE_HEADS.build(auxiliary_head)
        self.aux_loss_weight = aux_loss_weight

        self.train_cfg = train_cfg or {}
        self.test_cfg  = test_cfg  or {}

    # ── BaseSegmentor interface ───────────────────────────────────────────────

    def extract_feat(self, img):
        return self.backbone(img)

    def encode_decode(self, img, img_size=None):
        feats = self.extract_feat(img)
        return self.decode_head(feats, img_size=img_size)

    def forward_test(self, img):
        return self.encode_decode(img, img_size=(img.shape[2], img.shape[3]))

    def forward_train(self, img, gt_semantic_seg=None):
        """Training forward.

        If gt_semantic_seg is provided, computes and returns a loss dict.
        If None, returns raw logits — letting the caller handle the loss.

        Returns:
            dict {'loss_seg': tensor, ...}  when gt_semantic_seg is given
            tensor (B, C, H, W)             when gt_semantic_seg is None
        """
        feats    = self.extract_feat(img)
        img_size = (img.shape[2], img.shape[3])
        logits   = self.decode_head(feats, img_size=img_size)

        if gt_semantic_seg is None:
            return logits   # caller owns the loss

        losses = self._seg_loss(logits, gt_semantic_seg, name='loss_seg')

        if self.with_auxiliary_head:
            aux_logits = self.auxiliary_head(feats, img_size=img_size)
            for k, v in self._seg_loss(aux_logits, gt_semantic_seg, name='loss_aux').items():
                losses[k] = v * self.aux_loss_weight

        return losses

    def forward_both(
        self,
        img: torch.Tensor,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        """Trả về cả main logits và aux logits.
        
        Returns:
            main_out: (B, num_classes, H, W)
            aux_out:  (B, num_classes, H, W) hoặc None nếu không có aux head
        """
        img_size = img_size or (img.shape[2], img.shape[3])
        feats    = self.extract_feat(img)
        main_out = self.decode_head(feats, img_size=img_size)
        aux_out  = self.auxiliary_head(feats, img_size=img_size) \
                if self.with_auxiliary_head else None
        return main_out, aux_out
    # ── Built-in fallback loss (replace with LOSSES.build() later) ───────────

    @staticmethod
    def _seg_loss(logits, targets, ignore_index=255, name='loss_seg'):
        loss = F.cross_entropy(logits, targets.long(), ignore_index=ignore_index)
        return {name: loss}