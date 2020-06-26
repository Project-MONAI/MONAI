# Copyright 2020 MONAI Consortium
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
from typing import List

import torch
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Act, Conv, Norm
from torch import nn
from torchgpipe.skip import Namespace, pop, skippable, stash


@skippable(stash=["skip"], pop=[])
class Stash(nn.Module):
    def forward(self, input: torch.Tensor):
        yield stash("skip", input)
        return input  # noqa  using yield together with return


@skippable(stash=[], pop=["skip"])
class PopCat(nn.Module):
    def forward(self, input: torch.Tensor):
        skip = yield pop("skip")
        if skip is not None:
            input = torch.cat([input, skip], dim=1)
        return input


def flatten_sequential(module: nn.Sequential):
    """
    Recursively make all the submodules sequential.

    Args:
        module: a torch sequential model.
    """
    if not isinstance(module, nn.Sequential):
        raise TypeError("module must be a nn.Sequential instance.")

    def _flatten(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Sequential):
                for sub_name, sub_child in _flatten(child):
                    yield f"{name}_{sub_name}", sub_child
            else:
                yield name, child

    return nn.Sequential(OrderedDict(_flatten(module)))


class DoubleConv(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        stride=2,
        act_1=Act.LEAKYRELU,
        norm_1=Norm.BATCH,
        act_2=Act.LEAKYRELU,
        norm_2=Norm.BATCH,
        conv_only=True,
    ):
        """
        A sequence of Conv_1 + Norm_1 + Act_1 + Conv_2 (+ Norm_2 + Act_2).

        `norm_2` and `act_2` are ignored when `conv_only` is True.
        `stride` is for `Conv_1`, typically stride=2 for 2x spatial downsampling.

        Args:
            spatial_dims: number of the input spatial dimension.
            in_channels: number of input channels.
            out_channels: number of output channels.
            stride: stride of the first conv., mainly used for 2x downsampling when stride=2.
            act_1: activation type of the first convolution.
            norm_1: normalization type of the first convolution.
            act_2: activation type of the second convolution.
            norm_2: normalization type of the second convolution.
            conv_only: whether the second conv is convolution layer only. Default to True,
                indicates that `act_2` and `norm_2` are not in use.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            Convolution(spatial_dims, in_channels, out_channels, strides=stride, act=act_1, norm=norm_1, bias=False,),
            Convolution(spatial_dims, out_channels, out_channels, act=act_2, norm=norm_2, conv_only=conv_only),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPipe(nn.Sequential):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, n_feat: int = 32, depth: int = 4):
        """
        A UNet-like architecture for model parallelism.

        Args:
            spatial_dims: number of input spatial dimensions,
                2 for (B, in_channels, H, W), 3 for (B, in_channels, H, W, D).
            in_channels: number of input channels.
            out_channels: number of output channels.
            n_feat: number of features in the first convolution.
            depth: number of downsampling stages.
        """
        super(UNetPipe, self).__init__()
        n_enc_filter: List[int] = [n_feat]
        for i in range(1, depth + 1):
            n_enc_filter.append(min(n_enc_filter[-1] * 2, 1024))
        namespaces = [Namespace() for _ in range(depth)]

        # construct the encoder
        encoder_layers: List[nn.Module] = []
        init_conv = Convolution(
            spatial_dims, in_channels, n_enc_filter[0], strides=2, act=Act.LEAKYRELU, norm=Norm.BATCH, bias=False,
        )
        encoder_layers.append(
            nn.Sequential(OrderedDict([("Conv", init_conv,), ("skip", Stash().isolate(namespaces[0]))]))
        )
        for i in range(1, depth + 1):
            down_conv = DoubleConv(spatial_dims, n_enc_filter[i - 1], n_enc_filter[i])
            if i == depth:
                layer_dict = OrderedDict([("Down", down_conv)])
            else:
                layer_dict = OrderedDict([("Down", down_conv), ("skip", Stash().isolate(namespaces[i]))])
            encoder_layers.append(nn.Sequential(layer_dict))
        encoder = nn.Sequential(*encoder_layers)

        # construct the decoder
        decoder_layers: List[nn.Module] = []
        for i in reversed(range(1, depth + 1)):
            in_ch, out_ch = n_enc_filter[i], n_enc_filter[i - 1]
            layer_dict = OrderedDict(
                [
                    ("Up", UpSample(spatial_dims, in_ch, out_ch, 2, True)),
                    ("skip", PopCat().isolate(namespaces[i - 1])),
                    ("Conv1x1x1", Conv[Conv.CONV, spatial_dims](out_ch * 2, in_ch, kernel_size=1)),
                    ("Conv", DoubleConv(spatial_dims, in_ch, out_ch, stride=1, conv_only=True)),
                ]
            )
            decoder_layers.append(nn.Sequential(layer_dict))
        in_ch = min(n_enc_filter[0] // 2, 32)
        layer_dict = OrderedDict(
            [
                ("Up", UpSample(spatial_dims, n_feat, in_ch, 2, True)),
                ("RELU", Act[Act.LEAKYRELU](inplace=False)),
                ("out", Conv[Conv.CONV, spatial_dims](in_ch, out_channels, kernel_size=3, padding=1),),
            ]
        )
        decoder_layers.append(nn.Sequential(layer_dict))
        decoder = nn.Sequential(*decoder_layers)

        # making a sequential model
        self.add_module("encoder", encoder)
        self.add_module("decoder", decoder)

        for m in self.modules():
            if isinstance(m, Conv[Conv.CONV, spatial_dims]):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, Norm[Norm.BATCH, spatial_dims]):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Conv[Conv.CONVTRANS, spatial_dims]):
                nn.init.kaiming_normal_(m.weight)
