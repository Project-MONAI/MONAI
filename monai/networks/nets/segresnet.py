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

from typing import Dict, Optional

import torch.nn as nn

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_norm_layer
from monai.networks.blocks.upsample import UpSample as UpSample
from monai.networks.layers.factories import Act, Dropout


class SegResNet(nn.Module):
    """
    SegResNet:
    "3D MRI brain tumor segmentation using autoencoder regularization, https://arxiv.org/abs/1810.11654"
    The code is adapted from the author's code, the difference is that the variational autoencoder (VAE) 
    is not used during training in this implementation.
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        droupout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        norm_name: feature normalization type, this module only supports group norm, 
            batch norm and instance norm. Defaults to ``group``.
        num_groups: number of groups to separate the channels into. Defaults to 8.
        use_conv_final: if add a final convolution block to output an activated result. Defaults to True.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: The mode of upsampling manipulations, there are three choices:
            1) "transpose", uses transposed convolution layers.
            2) "bilinear", uses bilinear interpolate.
            3) "trilinear", uses trilinear interpolate.
            Using the last two modes cannot guarantee the model's reproducibility. Defaults to "trilinear".
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        norm_name: str = "group",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: str = "trilinear",
    ):
        super().__init__()

        assert spatial_dims == 2 or spatial_dims == 3, "spatial_dims can only be 2 or 3."

        if upsample_mode != "transpose":
            upsample_mode = "bilinear" if spatial_dims == 2 else "trilinear"

        resblock_params = {"norm_name": norm_name, "num_groups": num_groups}

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.upsample_mode = upsample_mode
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers(resblock_params)
        self.up_layers, self.up_samples = self._make_up_layers(resblock_params)

        if use_conv_final:
            self.conv_final = nn.Sequential(
                get_norm_layer(spatial_dims, init_filters, norm_name, num_groups=num_groups),
                Act[Act.RELU](inplace=True),
                get_conv_layer(spatial_dims, init_filters, out_channels, kernel_size=1, bias=True),
            )

        if dropout_prob:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self, resblock_params: Dict):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters = self.blocks_down, self.spatial_dims, self.init_filters
        for i in range(len(blocks_down)):
            layer_in_channels = filters * 2 ** i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, **resblock_params) for _ in range(blocks_down[i])]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self, resblock_params: Dict):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[ResBlock(spatial_dims, sample_in_channels // 2, **resblock_params) for _ in range(blocks_up[i])]
                )
            )
            up_module: nn.Module
            if upsample_mode == "transpose":
                up_module = UpSample(
                    spatial_dims,
                    filters * 2 ** (n_up - i - 1),
                    filters * 2 ** (n_up - i - 1),
                    scale_factor=2,
                    with_conv=True,
                )
            else:
                up_module = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False,)
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        up_module,
                    ]
                )
            )
        return up_layers, up_samples

    def forward(self, x):
        x = self.convInit(x)
        if self.dropout_prob:
            x = self.dropout(x)

        down_x = []
        for i in range(len(self.blocks_down)):
            x = self.down_layers[i](x)  # resblock
            down_x.append(x)
        down_x.reverse()

        for i in range(len(self.blocks_up)):
            x = self.up_samples[i](x) + down_x[i + 1]
            x = self.up_layers[i](x)
        if self.use_conv_final:
            return self.conv_final(x)
        return x
