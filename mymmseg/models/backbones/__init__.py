from .base_backbone import BaseBackbone
from .resnet import ResNet  # ← decorator @BACKBONES.register_module() chạy ở đây

__all__ = ["BaseBackbone", "ResNet"]