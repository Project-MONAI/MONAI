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

from collections import OrderedDict
from typing import Callable, Sequence, Type, Union, List

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks import UpSample
from monai.utils import InterpolateMode, UpsampleMode


__all__ = [
   "hovernet",
    "Hovernet",
    "HoVernet",
    "HoVerNet",
    "HoverNet",
]

class _DenseLayerDecoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        kernel_size: int = 3
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
            kernel_size: 3 or 5 depending on the 'mode' [fast or original]
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
        self.layers.add_module("conv2", conv_type(num_features, out_channels, kernel_size=kernel_size, padding=1, groups=4, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        new_features = self.layers(x)
        x = torch.cat([x, new_features], 1)
        x = x[:,:,1:-1,1:-1]

        return x

class _DecoderBlock(nn.Sequential):
    def __init__(
        self,
        layers: int,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            layers: number of layers in the block.
            num_features: number of internal features used.
            in_channels: number of the input channel.
            out_channels: number of the output channel.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            kernel_size: 3 or 5 depending on the 'mode' [fast or original]
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 2]

        self.add_module("conva", conv_type(in_channels, in_channels // 4, kernel_size=kernel_size, bias=False))

        _in_channels = in_channels // 4
        for i in range(layers):
            layer = _DenseLayerDecoder(num_features, _in_channels,
                                       out_channels, dropout_prob, act=act, norm=norm,kernel_size=kernel_size)
            _in_channels += out_channels
            self.add_module("denselayerdecoder%d" % (i + 1), layer)


        trans = _Transition(_in_channels, act=act, norm=norm)
        self.add_module(f"bna_block", trans)

        self.add_module("convf", conv_type(_in_channels, _in_channels, kernel_size=1, bias=False))

class _DenseLayer(nn.Sequential):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        out_channels: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        drop_first_norm_relu: int = 0,

    ) -> None:
        """
        Args:
            num_features: number of internal channels used for the layer
            in_channels: number of the input channels.
            out_channels: number of the output channels.
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
            drop_first_norm_relu - omits the first norm/relu for the first layer
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 2]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, 2]

        if not drop_first_norm_relu:
            self.add_module("preact_norm", get_norm_layer(name=norm, spatial_dims=2, channels=in_channels))
            self.add_module("preact_relu", get_act_layer(name=act))

        self.add_module("conv1", conv_type(in_channels, num_features, kernel_size=1, padding=0, bias=False))

        self.add_module("norm2", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.add_module("relu2", get_act_layer(name=act))

        if in_channels!=64 and drop_first_norm_relu:
            self.add_module("conv2", conv_type(num_features, num_features, kernel_size=3, stride=2, padding=2, bias=False))
        else:
            self.add_module("conv2", conv_type(num_features, num_features, kernel_size=3, padding=1, bias=False))

        self.add_module("norm3", get_norm_layer(name=norm, spatial_dims=2, channels=num_features))
        self.add_module("relu3", get_act_layer(name=act))
        self.add_module("conv3", conv_type(num_features, out_channels, kernel_size=1, padding=0, bias=False))

        if dropout_prob > 0:
            self.add_module("dropout", dropout_type(dropout_prob))


class _Transition(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
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
        dropout_prob: float = 0,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
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

        if in_channels==64:
            self.shortcut = conv_type(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = conv_type(in_channels, out_channels, kernel_size=1, stride=2, padding=1, bias=False)

        layer = _DenseLayer(num_features, in_channels, out_channels, dropout_prob,
                                act=act, norm=norm, drop_first_norm_relu=True)
        self.layers.add_module("prim_denselayer%d" % (1), layer)

        for i in range(1, layers):
            layer = _DenseLayer(num_features, out_channels, out_channels, dropout_prob, act=act, norm=norm)
            self.layers.add_module("main_denselayer%d" % (i + 1), layer)

        self.bna_block = _Transition(out_channels, act=act, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        s = x.shape
        sc = self.shortcut(x)

        if s[-1] != sc.shape[-1]:
            sc = sc[:,:,:-1,:-1]

        for layer in self.layers:
            x = layer.forward(x)

            if x.shape[-1] != sc.shape[-1]:
                x = x[:,:,:-1,:-1]

            x = x + sc
            sc = x

        x = self.bna_block(x)

        return x

class _DecoderBranch(nn.ModuleList):
    def __init__(self,
        decode_config: Sequence[int] = (8, 4),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
        out_channels: int = 2,
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
        decode_config: number of layers in each block.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
        out_channels: number of classes to predict for exach pixel
        kernel_size: 3 or 5 depending on the 'mode' [fast or original]
        """
        super().__init__()
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, 2]

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
        i = len(decode_config)
        block = nn.Sequential(
            OrderedDict([ ("conva", conv_type(256, 64, kernel_size=kernel_size, stride=1, bias=False, padding=1)) ])
        )

        self.output_features.add_module(f"decoderblock{i + 1}", block)

        block = nn.Sequential(
            OrderedDict([
                ("norm", get_norm_layer(name=norm, spatial_dims=2, channels=64)),
                ("relu", get_act_layer(name=act)),
                ("conv", conv_type(64, out_channels, kernel_size=1, stride=1))
            ])
        )

        self.output_features.add_module(f"decoderblock{i + 2}", block)

        self.upsample = UpSample(2, scale_factor=2, mode=UpsampleMode.NONTRAINABLE,
                                        interp_mode=InterpolateMode.BILINEAR, bias=False)

    def forward(self, x: torch.Tensor, short_cuts: List[torch.Tensor]) -> torch.Tensor:

        block_number=len(short_cuts)-1

        for block in self.decoder_blocks:
            x += short_cuts[block_number]
            x = block(x)
            x = self.upsample(x)
            block_number-=1

        for block in self.output_features:
            if block_number>=0:
                xp = short_cuts[block_number]
                x += xp

            x = block(x)
            block_number-=1

        return x

class HoverNet(nn.Module):
    """
     Graham S, Vu QD, Raza SEA, Azam A, Tsang YW, Kwak JT, Rajpoot N.
     Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
     Med Image Anal. 2019 Dec;58:101563.
     doi: 10.1016/j.media.2019.101563.
     Epub 2019 Sep 18. PMID: 31561183.

    Args:
        in_channels: number of the input channel.
        out_classes: number of the nuclei classes.
        init_features: number of filters in the first convolution layer.
        block_config: how many layers in each residual block.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
        mode: 'original' or 'fast' - larger or smaller kernel size for decoder
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_classes: int = None,
        init_features: int = 64,
        block_config: Sequence[int] = (3, 4, 6, 3),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.2,
        mode: str = "fast",
    ) -> None:

        super().__init__()

        self.kernel_size = 3 if mode == "fast" else 5
        self.out_classes = out_classes

        conv_type: nn.Conv2d = Conv[Conv.CONV, 2]

        print("In_Channels =", in_channels)
        self.input_features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=1, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=2, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                ]
            )
        )

        _in_channels = init_features
        _out_channels = 256
        _num_features = init_features

        self.res_blocks = nn.Sequential()

        for i, num_layers in enumerate(block_config):
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
        self.bottleneck.add_module("conv_bottleneck", conv_type(_in_channels, _num_features, kernel_size=1, stride=1, padding=0, bias=False))

        self.upsample = UpSample(2, scale_factor=2, mode=UpsampleMode.NONTRAINABLE,
                                           interp_mode=InterpolateMode.BILINEAR, bias=False)

        # decode branches
        self.nucleus_prediction = _DecoderBranch(kernel_size=self.kernel_size)
        self.horizontal_vertical = _DecoderBranch(kernel_size=self.kernel_size)

        if out_classes:
            self.type_prediction = _DecoderBranch(out_channels=out_classes, kernel_size=self.kernel_size)
        else:
            self.type_prediction = None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        imgs = x / 255.0  # to 0-1 range to match XY
        x = self.input_features(x)
        num_blocks = len(self.res_blocks)
        short_cuts = []

        for i, block in enumerate(self.res_blocks):
            x = block.forward(x)

            if i==0:
                short_cuts.append(x[:,:,46:-46,46:-46])
            elif i==1:
                short_cuts.append(x[:,:,18:-18,18:-18])
            elif i==2:
                short_cuts.append(x)

        x = self.bottleneck(x)
        x = self.upsample(x)

        x_np = torch.tensor(x)
        x_np = self.nucleus_prediction(x_np, short_cuts)

        x_hv = torch.tensor(x)
        x_hv = self.horizontal_vertical(x_hv, short_cuts)


        if self.type_prediction:
            x_tp = torch.tensor(x)
            x_tp = self.type_prediction(x_tp, short_cuts)
            return {"type_prediction": x_tp, "nucleus_prediction": x_np, "horizonal_vertical": x_hv}

        return {"nucleus_prediction": x_np, "horizonal_vertical": x_hv}


hovernet = Hovernet = HoVernet = HoVerNet = HoverNet
