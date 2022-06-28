# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# Adapted from https://github.com/vqdang/hover_net
# which has the following license:
# https://github.com/vqdang/hover_net/blob/master/LICENSE
# MIT License

# Origial publication:
#  @article{graham2019hover,
#    title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
#    author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak,
#            Jin Tae and Rajpoot, Nasir},
#    journal={Medical Image Analysis},
#    pages={101563},
#    year={2019},
#    publisher={Elsevier}
# }

# =========================================================================

from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, List, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import UpSample
from monai.networks.layers.factories import Conv, Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import InterpolateMode, UpsampleMode, export

__all__ = ["HoverNet", "hovernet", "Hovernet", "HoVernet", "HoVerNet"]


class _DenseLayerDecoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            num_features: number of internal channels used for the layer
            in_channels: number of the input channels.
            out_channels: number of the output channels.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 2]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, 2]

        self.layers = nn.Sequential()

        self.layers.add_module("preact_bna/bn", get_norm_layer(name=norm, spatial_dims=2, channels=in_channels))
        self.layers.add_module("preact_bna/relu", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, num_features, kernel_size=1, bias=False))
        self.layers.add_module("conv1/norm", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.layers.add_module("conv1/relu2", get_act_layer(name=act))
        self.layers.add_module(
            "conv2", conv_type(num_features, out_channels, kernel_size=kernel_size, padding=0, groups=4, bias=False)
        )

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.layers(x)
        if x1.shape != x.shape:
            trim = (x.shape[-1] - x1.shape[-1]) // 2
            x = x[:, :, trim:-trim, trim:-trim]

        x = torch.cat([x, x1], 1)

        return x


class _DecoderBlock(nn.Sequential):
    def __init__(
        self,
        layers: int,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            num_features: number of internal features used.
            in_channels: number of the input channel.
            out_channels: number of the output channel.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 2]

        self.add_module("conva", conv_type(in_channels, in_channels // 4, kernel_size=kernel_size, bias=False))

        _in_channels = in_channels // 4
        for i in range(layers):
            layer = _DenseLayerDecoder(
                num_features, _in_channels, out_channels, dropout_prob, act=act, norm=norm, kernel_size=kernel_size
            )
            _in_channels += out_channels
            self.add_module("denselayerdecoder%d" % (i + 1), layer)

        trans = _Transition(_in_channels, act=act, norm=norm)
        self.add_module("bna_block", trans)
        self.add_module("convf", conv_type(_in_channels, _in_channels, kernel_size=1, bias=False))


class _DenseLayer(nn.Sequential):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        drop_first_norm_relu: int = 0,
        kernel_size: int = 3,
    ) -> None:
        """Dense Convolutional Block.

        References:
            Huang, Gao, et al. "Densely connected convolutional networks."
            Proceedings of the IEEE conference on computer vision and
            pattern recognition. 2017.

        Args:
            num_features: number of internal channels used for the layer
            in_channels: number of the input channels.
            out_channels: number of the output channels.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            drop_first_norm_relu - omits the first norm/relu for the first layer
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
        """
        super().__init__()

        self.layers = nn.Sequential()
        conv_type: Callable = Conv[Conv.CONV, 2]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, 2]

        if not drop_first_norm_relu:
            self.layers.add_module("preact_norm", get_norm_layer(name=norm, spatial_dims=2, channels=in_channels))
            self.layers.add_module("preact_relu", get_act_layer(name=act))

        self.layers.add_module("conv1", conv_type(in_channels, num_features, kernel_size=1, padding=0, bias=False))
        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.layers.add_module("relu2", get_act_layer(name=act))

        if in_channels != 64 and drop_first_norm_relu:
            self.layers.add_module(
                "conv2", conv_type(num_features, num_features, kernel_size=kernel_size, stride=2, padding=2, bias=False)
            )
        else:
            self.layers.add_module("conv2", conv_type(num_features, num_features, kernel_size=1, padding=0, bias=False))

        self.layers.add_module("norm3", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.layers.add_module("relu3", get_act_layer(name=act))
        self.layers.add_module("conv3", conv_type(num_features, out_channels, kernel_size=1, padding=0, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))


class _Transition(nn.Sequential):
    def __init__(
        self, in_channels: int, act: Union[str, tuple] = ("relu", {"inplace": True}), norm: Union[str, tuple] = "batch"
    ) -> None:
        """
        Args:
            in_channels: number of the input channel.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=2, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """Residual block.

        References:
            He, Kaiming, et al. "Deep residual learning for image
            recognition." Proceedings of the IEEE conference on computer
            vision and pattern recognition. 2016.

        Args:
            layers: number of layers in the block.
            num_features: number of internal features used.
            in_channels: number of the input channel.
            out_channels: number of the output channel.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        self.layers = nn.Sequential()
        conv_type: Callable = Conv[Conv.CONV, 2]

        if in_channels == 64:
            self.shortcut = conv_type(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = conv_type(in_channels, out_channels, kernel_size=1, stride=2, padding=1, bias=False)

        layer = _DenseLayer(
            num_features, in_channels, out_channels, dropout_prob, act=act, norm=norm, drop_first_norm_relu=True
        )
        self.layers.add_module("prim_denselayer%d" % (1), layer)

        for i in range(1, layers):
            layer = _DenseLayer(num_features, out_channels, out_channels, dropout_prob, act=act, norm=norm)
            self.layers.add_module("main_denselayer%d" % (i + 1), layer)

        self.bna_block = _Transition(out_channels, act=act, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        sc = self.shortcut(x)

        if self.shortcut.stride == (2, 2):
            sc = sc[:, :, :-1, :-1]

        for layer in self.layers:
            x = layer.forward(x)
            if x.shape[-2:] != sc.shape[-2:]:
                x = x[:, :, :-1, :-1]

            x = x + sc
            sc = x

        x = self.bna_block(x)

        return x


class _DecoderBranch(nn.ModuleList):
    def __init__(
        self,
        decode_config: Sequence[int] = (8, 4),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
        out_channels: int = 2,
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            decode_config: number of layers for each block.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            dropout_prob: dropout rate after each dense layer.
            num_features: number of internal features used.
            out_channels: number of the output channel.
            kernel_size: size of the kernel for >1 convolutions (dependent on mode)
        """
        super().__init__()
        conv_type: Callable = Conv[Conv.CONV, 2]

        # decode branches
        _in_channels = 1024
        _num_features = 128
        _out_channels = 32

        self.decoder_blocks = nn.Sequential()
        for i, num_layers in enumerate(decode_config):
            block = _DecoderBlock(
                layers=num_layers,
                num_features=_num_features,
                in_channels=_in_channels,
                out_channels=_out_channels,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
                kernel_size=kernel_size,
            )
            self.decoder_blocks.add_module(f"decoderblock{i + 1}", block)
            _in_channels = 512

        # output layers
        self.output_features = nn.Sequential()
        _i = len(decode_config)
        _pad_size = (kernel_size - 1) // 2
        block = nn.Sequential(
            OrderedDict(
                [("conva", conv_type(256, 64, kernel_size=kernel_size, stride=1, bias=False, padding=_pad_size))]
            )
        )

        self.output_features.add_module(f"decoderblock{_i + 1}", block)

        block = nn.Sequential(
            OrderedDict(
                [
                    ("norm", get_norm_layer(name=norm, spatial_dims=2, channels=64)),
                    ("relu", get_act_layer(name=act)),
                    ("conv", conv_type(64, out_channels, kernel_size=1, stride=1)),
                ]
            )
        )

        self.output_features.add_module(f"decoderblock{_i + 2}", block)

        self.upsample = UpSample(
            2, scale_factor=2, mode=UpsampleMode.NONTRAINABLE, interp_mode=InterpolateMode.BILINEAR, bias=False
        )

    def forward(self, xin: torch.Tensor, short_cuts: List[torch.Tensor]) -> torch.Tensor:

        block_number = len(short_cuts) - 1
        x = xin + short_cuts[block_number]

        for block in self.decoder_blocks:
            x = block(x)
            x = self.upsample(x)
            block_number -= 1
            trim = (short_cuts[block_number].shape[-1] - x.shape[-1]) // 2
            x += short_cuts[block_number][:, :, trim:-trim, trim:-trim]

        for block in self.output_features:
            x = block(x)

        return x


@export("monai.networks.nets")
class HoverNet(nn.Module):
    """HoVerNet

    References:
      Graham, Simon et al. Hover-net: Simultaneous segmentation
      and classification of nuclei in multi-tissue histology images,
      Medical Image Analysis 2019

    Args:
        in_channels: number of the input channel.
        out_classes: number of the nuclear type classes.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    class Mode(Enum):
        FAST: int = 0
        ORIGINAL: int = 1

    def _mode_to_int(self, mode) -> int:

        if mode == self.Mode.FAST:
            return 0
        else:
            return 1

    def __init__(
        self,
        mode: int = Mode.FAST,
        in_channels: int = 3,
        out_classes: int = None,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        self.mode: int = self._mode_to_int(mode)

        if mode not in [self.Mode.ORIGINAL, self.Mode.FAST]:
            raise ValueError("Input size should be 270 x 270 when using Mode.ORIGINAL")

        if out_classes == 0:
            out_classes = None

        if out_classes:
            if out_classes > 128:
                raise ValueError("Number of nuclear types classes exceeds maximum (128)")
            elif out_classes == 1:
                raise ValueError("Number of nuclear type classes should either be None or >1")

        if dropout_prob > 1 or dropout_prob < 0:
            raise ValueError("Dropout can only be in the range 0.0 to 1.0")

        # number of filters in the first convolution layer.
        _init_features: int = 64
        # number of layers in each pooling block.
        _block_config: Sequence[int] = (3, 4, 6, 3)

        if mode == self.Mode.FAST:
            _ksize = 3
            _pad = 3
        else:
            _ksize = 5
            _pad = 0

        conv_type: Callable = Conv[Conv.CONV, 2]

        self.input_features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        conv_type(in_channels, _init_features, kernel_size=7, stride=1, padding=_pad, bias=False),
                    ),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=2, channels=_init_features)),
                    ("relu0", get_act_layer(name=act)),
                ]
            )
        )

        _in_channels = _init_features
        _out_channels = 256
        _num_features = _init_features

        self.res_blocks = nn.Sequential()

        for i, num_layers in enumerate(_block_config):
            block = _ResidualBlock(
                layers=num_layers,
                num_features=_num_features,
                in_channels=_in_channels,
                out_channels=_out_channels,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.res_blocks.add_module(f"residualblock{i + 1}", block)

            _in_channels = _out_channels
            _out_channels *= 2
            _num_features *= 2

        # bottleneck convolution
        self.bottleneck = nn.Sequential()
        self.bottleneck.add_module(
            "conv_bottleneck", conv_type(_in_channels, _num_features, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.upsample = UpSample(
            2, scale_factor=2, mode=UpsampleMode.NONTRAINABLE, interp_mode=InterpolateMode.BILINEAR, bias=False
        )

        # decode branches
        self.nucleus_prediction = _DecoderBranch(kernel_size=_ksize)
        self.horizontal_vertical = _DecoderBranch(kernel_size=_ksize)

        if out_classes:
            self.type_prediction = _DecoderBranch(out_channels=out_classes, kernel_size=_ksize)
        else:
            self.type_prediction = None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        if self.mode == 1:
            if x.shape[-1] != 270 or x.shape[-2] != 270:
                raise ValueError("Input size should be 270 x 270 when using Mode.ORIGINAL")
        else:
            if x.shape[-1] != 256 or x.shape[-2] != 256:
                raise ValueError("Input size should be 256 x 256 when using Mode.FAST")

        x = x / 255.0  # to 0-1 range to match XY
        x = self.input_features(x)
        short_cuts = []

        for i, block in enumerate(self.res_blocks):
            x = block.forward(x)

            if i <= 2:
                short_cuts.append(x)

        x = self.bottleneck(x)
        x = self.upsample(x)

        x_np = self.nucleus_prediction(x, short_cuts)
        x_hv = self.horizontal_vertical(x, short_cuts)

        if self.type_prediction is not None:
            x_tp = self.type_prediction(x, short_cuts)
            return {"nucleus_prediction": x_np, "horizonal_vertical": x_hv, "type_prediction": x_tp}

        return {"nucleus_prediction": x_np, "horizonal_vertical": x_hv}


hovernet = Hovernet = HoVernet = HoVerNet = HoverNet
