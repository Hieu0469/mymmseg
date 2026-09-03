"""ResNet backbone — API-compatible with mmseg's ResNet.

Self-contained: no torchvision, no mmcv, no mmengine deps.
Supports BasicBlock / Bottleneck, deep_stem, avg_down, dilation,
multi_grid, DCN placeholder, frozen_stages, norm_eval.
"""

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from ...utils.registry import BACKBONES
from ...utils.base_module import BaseModule


# ── Norm helper ───────────────────────────────────────────────────────────────

def build_norm_layer(norm_cfg: dict, num_channels: int, postfix: Union[int, str] = ''):
    """Returns (name, layer) like mmcv.build_norm_layer."""
    cfg = norm_cfg.copy()
    norm_type = cfg.pop('type', 'BN')
    requires_grad = cfg.pop('requires_grad', True)

    if norm_type == 'BN':
        layer = nn.BatchNorm2d(num_channels, **cfg)
    elif norm_type == 'GN':
        layer = nn.GroupNorm(num_channels=num_channels, **cfg)
    elif norm_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_channels, **cfg)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

    for p in layer.parameters():
        p.requires_grad = requires_grad

    abbr = norm_type.lower()
    name = abbr + str(postfix)
    return name, layer


def build_conv_layer(conv_cfg: Optional[dict], *args, **kwargs) -> nn.Conv2d:
    """Returns a Conv2d (conv_cfg=None → standard Conv2d, matching mmcv API)."""
    if conv_cfg is None or conv_cfg.get('type') == 'Conv2d':
        return nn.Conv2d(*args, **kwargs)
    raise NotImplementedError(f"conv_cfg type '{conv_cfg['type']}' not yet supported.")


# ── ResLayer ──────────────────────────────────────────────────────────────────

class ResLayer(nn.Sequential):
    """Pack multiple blocks into one stage (mirrors mmseg's ResLayer)."""

    def __init__(
        self,
        block,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        dilation: int = 1,
        avg_down: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type='BN'),
        multi_grid: Optional[Sequence[int]] = None,
        contract_dilation: bool = False,
        with_cp: bool = False,
        dcn: Optional[dict] = None,
        plugins: Optional[list] = None,
        style: str = 'pytorch',
        downsample: Optional[nn.Module] = None,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ):
        self.block = block
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample_layers = []
            if avg_down:
                if stride == 1:
                    downsample_layers.append(nn.AvgPool2d(1, stride=1))
                else:
                    downsample_layers.append(nn.AvgPool2d(stride, stride=stride, ceil_mode=True))
                downsample_layers += [
                    build_conv_layer(conv_cfg, inplanes, planes * block.expansion,
                                     kernel_size=1, stride=1, bias=False),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1],
                ]
            else:
                downsample_layers = [
                    build_conv_layer(conv_cfg, inplanes, planes * block.expansion,
                                     kernel_size=1, stride=stride, bias=False),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1],
                ]
            downsample = nn.Sequential(*downsample_layers)

        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=plugins,
                init_cfg=init_cfg,
            )
        )
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            _dilation = dilation if multi_grid is None else multi_grid[i % len(multi_grid)]
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=_dilation,
                    style=style,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    plugins=plugins,
                    init_cfg=init_cfg,
                )
            )

        super().__init__(*layers)


# ── Blocks ────────────────────────────────────────────────────────────────────

class BasicBlock(BaseModule):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        style: str = 'pytorch',
        with_cp: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type='BN'),
        dcn: Optional[dict] = None,
        plugins=None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        assert dcn is None, 'DCN not implemented for BasicBlock'
        assert plugins is None, 'Plugins not implemented for BasicBlock'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, 3,
            stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self): return getattr(self, self.norm1_name)
    @property
    def norm2(self): return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner(x):
            identity = x
            out = self.relu(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return out + identity

        out = cp.checkpoint(_inner, x) if self.with_cp and x.requires_grad else _inner(x)
        return self.relu(out)


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        style: str = 'pytorch',
        with_cp: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type='BN'),
        dcn: Optional[dict] = None,
        plugins=None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        assert style in ('pytorch', 'caffe')
        assert plugins is None, 'Plugins not yet supported'

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.dcn = dcn
        self.with_dcn = dcn is not None

        conv1_stride, conv2_stride = (1, stride) if style == 'pytorch' else (stride, 1)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, 1, stride=conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)

        if self.with_dcn:
            # placeholder — swap in your DCN impl here
            raise NotImplementedError("DCN support: replace this with your DCN conv")
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3,
            stride=conv2_stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.conv3 = build_conv_layer(conv_cfg, planes, planes * self.expansion, 1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self): return getattr(self, self.norm1_name)
    @property
    def norm2(self): return getattr(self, self.norm2_name)
    @property
    def norm3(self): return getattr(self, self.norm3_name)

    def forward(self, x):
        def _inner(x):
            identity = x
            out = self.relu(self.norm1(self.conv1(x)))
            out = self.relu(self.norm2(self.conv2(out)))
            out = self.norm3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return out + identity

        out = cp.checkpoint(_inner, x) if self.with_cp and x.requires_grad else _inner(x)
        return self.relu(out)


# ── ResNet ────────────────────────────────────────────────────────────────────

@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone — API-compatible with mmseg's ResNet.

    Args:
        depth (int): 18 | 34 | 50 | 101 | 152
        in_channels (int): Input channels. Default: 3.
        stem_channels (int): Stem output channels. Default: 64.
        base_channels (int): Stage-0 channels before expansion. Default: 64.
        num_stages (int): Number of stages. Default: 4.
        strides: Stride of first block per stage. Default: (1,2,2,2).
        dilations: Dilation per stage. Default: (1,1,1,1).
        out_indices: Which stages to output. Default: (0,1,2,3).
        style: 'pytorch' or 'caffe'. Default: 'pytorch'.
        deep_stem: Replace 7×7 stem with 3× 3×3. Default: False.
        avg_down: AvgPool in downsampling shortcut. Default: False.
        frozen_stages: Freeze stem + first N stages. Default: -1.
        norm_cfg: Norm layer config. Default: dict(type='BN').
        norm_eval: Keep BN in eval mode during training. Default: False.
        with_cp: Gradient checkpointing. Default: False.
        zero_init_residual: Zero-init last BN in each block. Default: True.
        multi_grid: Multi-grid dilation for last stage. Default: None.
        contract_dilation: Halve first dilation in each stage. Default: False.
        pretrained (str): Path to pretrained weights. Default: None.
        init_cfg: Init config dict. Default: None.
    """

    arch_settings = {
        18:  (BasicBlock,  (2, 2, 2, 2)),
        34:  (BasicBlock,  (3, 4, 6, 3)),
        50:  (Bottleneck,  (3, 4, 6, 3)),
        101: (Bottleneck,  (3, 4, 23, 3)),
        152: (Bottleneck,  (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth: int,
        in_channels: int = 3,
        stem_channels: int = 64,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        style: str = 'pytorch',
        deep_stem: bool = False,
        avg_down: bool = False,
        frozen_stages: int = -1,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        norm_eval: bool = False,
        dcn: Optional[dict] = None,
        stage_with_dcn: Sequence[bool] = (False, False, False, False),
        plugins=None,
        multi_grid: Optional[Sequence[int]] = None,
        contract_dilation: bool = False,
        with_cp: bool = False,
        zero_init_residual: bool = True,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[dict, List[dict]]] = None,
    ):
        super().__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot both be set'

        if isinstance(pretrained, str):
            warnings.warn('pretrained is deprecated — use init_cfg instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None and init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer=['BatchNorm2d']),
            ]
        elif pretrained is not None:
            raise TypeError('pretrained must be str or None')

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.plugins = plugins
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        assert 1 <= num_stages <= 4
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        # zero-init last BN per block
        if zero_init_residual:
            norm_name = 'norm2' if self.block is BasicBlock else 'norm3'
            block_init_cfg = dict(type='Constant', val=0, override=dict(name=norm_name))
        else:
            block_init_cfg = None

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            _dcn = self.dcn if (dcn is not None and stage_with_dcn[i]) else None
            stage_multi_grid = multi_grid if i == len(self.stage_blocks) - 1 else None
            planes = base_channels * 2 ** i

            res_layer = ResLayer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=strides[i],
                dilation=dilations[i],
                style=style,
                avg_down=avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=_dcn,
                plugins=self.make_stage_plugins(plugins, i) if plugins else None,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def make_stage_plugins(self, plugins, stage_idx):
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)
        return stage_plugins

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(self.conv_cfg, in_channels, stem_channels // 2,
                                 3, stride=2, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(self.conv_cfg, stem_channels // 2, stem_channels // 2,
                                 3, stride=1, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(self.conv_cfg, stem_channels // 2, stem_channels,
                                 3, stride=1, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg, in_channels, stem_channels,
                7, stride=2, padding=3, bias=False)
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for p in self.stem.parameters():
                    p.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for p in m.parameters():
                        p.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.relu(self.norm1(self.conv1(x)))
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        return self


# ── Variants (same as mmseg) ──────────────────────────────────────────────────

@BACKBONES.register_module()
class ResNetV1c(ResNet):
    """ResNetV1c: deep_stem=True, avg_down=False."""
    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    """ResNetV1d: deep_stem=True, avg_down=True."""
    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=True, **kwargs)