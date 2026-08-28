from .base_decode_head import BaseDecodeHead
from .fcn_head import FCNHead  # ← decorator @DECODE_HEADS.register_module() chạy ở đây

__all__ = ["BaseDecodeHead", "FCNHead"]