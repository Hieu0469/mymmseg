"""Smoke test — verify the semseg library wires together correctly."""

import sys
sys.path.insert(0, "/home/claude")

import torch
import semseg
from semseg.models.segmentors import EncoderDecoder

print(f"semseg version: {semseg.__version__}")
print(f"Registries — BACKBONES: {semseg.BACKBONES}")
print(f"Registries — DECODE_HEADS: {semseg.DECODE_HEADS}")
print(f"Registries — SEGMENTORS: {semseg.SEGMENTORS}")
print()

# ── Build from config dicts (MMSeg-style) ──────────────────────────────────
model = EncoderDecoder(
    backbone=dict(
        type="ResNet",
        depth=50,
        output_stride=16,       # dilated, effective stride 16
        out_indices=(0, 1, 2, 3),
        pretrained=False,       # skip download in CI
    ),
    decode_head=dict(
        type="FCNHead",
        in_channels=2048,
        in_index=-1,            # use last backbone output
        channels=256,
        num_classes=19,         # Cityscapes
        num_convs=2,
        dropout_ratio=0.1,
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,             # use layer3 output
        channels=256,
        num_classes=19,
        num_convs=1,
        dropout_ratio=0.1,
    ),
    aux_loss_weight=0.4,
)

print(f"Model type: {type(model).__name__}")
print(f"Backbone out_channels: {model.backbone.out_channels}")
print()

# ── Inference (eval mode) ──────────────────────────────────────────────────
model.eval()
img = torch.randn(2, 3, 512, 512)
with torch.no_grad():
    logits = model(img)
print(f"[INFERENCE] input: {tuple(img.shape)} → logits: {tuple(logits.shape)}")
assert logits.shape == (2, 19, 512, 512), f"Shape mismatch: {logits.shape}"

# ── Training (train mode) ──────────────────────────────────────────────────
model.train()
gt = torch.randint(0, 19, (2, 512, 512))
losses = model(img, gt_semantic_seg=gt)
print(f"[TRAINING]  loss keys: {list(losses.keys())}")
for k, v in losses.items():
    print(f"            {k}: {v.item():.4f}")

total_loss = sum(losses.values())
total_loss.backward()
print()
print("✓ All checks passed.")
