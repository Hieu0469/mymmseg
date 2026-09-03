"""BaseModule — lightweight port of mmengine.model.BaseModule.

Stripped of mmengine/mmcv dependencies so the library is self-contained.
API-compatible with the original: supports init_cfg, init_weights(), is_init.
"""

import copy
from abc import ABCMeta
from typing import Iterable, List, Optional, Union

import torch.nn as nn


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base class for all modules in this library.

    Mirrors mmengine.model.BaseModule but without mmengine/mmcv deps.

    Args:
        init_cfg: Initialization config dict or list of dicts.
            Supported types: 'Kaiming', 'Xavier', 'Constant', 'Normal',
            'Uniform', 'Pretrained'.
    """

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self) -> bool:
        return self._is_init

    @is_init.setter
    def is_init(self, value: bool):
        self._is_init = value

    def init_weights(self):
        """Initialize weights according to self.init_cfg, then recurse."""
        if self._is_init:
            return

        if self.init_cfg:
            cfgs = self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]
            # Pretrained last (highest priority — overwrites everything)
            pretrained = [c for c in cfgs if c.get('type') == 'Pretrained']
            others     = [c for c in cfgs if c.get('type') != 'Pretrained']
            for cfg in others:
                _apply_init_cfg(self, cfg)

        # Recurse into children
        for m in self.children():
            if hasattr(m, 'init_weights') and not getattr(m, 'is_init', False):
                m.init_weights()

        if self.init_cfg:
            for cfg in pretrained:
                _apply_init_cfg(self, cfg)

        self._is_init = True

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


# ── init_cfg dispatch ─────────────────────────────────────────────────────────

def _apply_init_cfg(module: nn.Module, cfg: dict):
    cfg = cfg.copy()
    init_type = cfg.pop('type')
    override  = cfg.pop('override', None)  # e.g. dict(name='norm3')

    if init_type == 'Pretrained':
        _load_pretrained(module, cfg)
    elif init_type == 'Kaiming':
        _init_by_type(module, cfg, override, _kaiming_init)
    elif init_type == 'Xavier':
        _init_by_type(module, cfg, override, _xavier_init)
    elif init_type == 'Constant':
        _init_by_type(module, cfg, override, _constant_init)
    elif init_type == 'Normal':
        _init_by_type(module, cfg, override, _normal_init)
    elif init_type == 'Uniform':
        _init_by_type(module, cfg, override, _uniform_init)
    else:
        raise ValueError(f"Unknown init type: '{init_type}'")


def _init_by_type(module, cfg, override, fn):
    """Apply fn to matched layers, or to a named submodule if override given."""
    layer_types = cfg.pop('layer', None)  # e.g. 'Conv2d' or ['Conv2d', 'Linear']
    if layer_types and not isinstance(layer_types, (list, tuple)):
        layer_types = [layer_types]

    if override:
        # Target a specific named sub-module
        name = override.get('name')
        sub = dict(module.named_modules()).get(name)
        if sub is not None:
            fn(sub, **{**cfg, **{k: v for k, v in override.items() if k != 'name'}})
        return

    for m in module.modules():
        if layer_types:
            if type(m).__name__ in layer_types:
                fn(m, **cfg)
        else:
            fn(m, **cfg)


# ── per-layer init functions ──────────────────────────────────────────────────

def _kaiming_init(m, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal', **_):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias)

def _xavier_init(m, gain=1, bias=0, distribution='normal', **_):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if distribution == 'uniform':
            nn.init.xavier_uniform_(m.weight, gain=gain)
        else:
            nn.init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias)

def _constant_init(m, val, bias=0, **_):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.constant_(m.weight, val)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, bias)

def _normal_init(m, mean=0, std=1, bias=0, **_):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias)

def _uniform_init(m, a=0, b=1, bias=0, **_):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=a, b=b)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias)

def _load_pretrained(module, cfg):
    import torch
    checkpoint = cfg.get('checkpoint')
    if checkpoint is None:
        return
    print(f"[BaseModule] Loading pretrained weights from: {checkpoint}")
    state = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys : {missing}")
    if unexpected:
        print(f"  Unexpected   : {unexpected}")


# ── MMSeg-compatible container classes ───────────────────────────────────────

class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    def __init__(self, modules: Optional[Iterable] = None, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class ModuleDict(BaseModule, nn.ModuleDict):
    def __init__(self, modules: Optional[dict] = None, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)