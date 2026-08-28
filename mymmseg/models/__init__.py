# Import subpackages → trigger tất cả __init__.py con
from . import backbones      # ← chạy backbones/__init__.py → ResNet registered
from . import decode_heads   # ← chạy decode_heads/__init__.py → FCNHead registered
from . import segmentors

from .backbones import BaseBackbone, ResNet
from .decode_heads import BaseDecodeHead, FCNHead
from .segmentors import BaseSegmentor, EncoderDecoder