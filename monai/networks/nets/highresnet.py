# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import ADN, Convolution
from monai.networks.layers.simplelayers import ChannelPad
from monai.utils import ChannelMatching

DEFAULT_LAYER_PARAMS_3D = (
    # initial conv layer
    {"name": "conv_0", "n_features": 16, "kernel_size": 3},
    # residual blocks
    {"name": "res_1", "n_features": 16, "kernels": (3, 3), "repeat": 3},
    {"name": "res_2", "n_features": 32, "kernels": (3, 3), "repeat": 3},
    {"name": "res_3", "n_features": 64, "kernels": (3, 3), "repeat": 3},
    # final conv layers
    {"name": "conv_1", "n_features": 80, "kernel_size": 1},
    {"name": "conv_2", "kernel_size": 1},
)


class HighResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernels: Sequence[int] = (3, 3),
        dilation: Union[Sequence[int], int] = 1,
        norm_type: Union[Tuple, str] = ("batch", {"affine": True}),
        acti_type: Union[Tuple, str] = ("relu", {"inplace": True}),
        channel_matching: Union[ChannelMatching, str] = ChannelMatching.PAD,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernels: each integer k in `kernels` corresponds to a convolution layer with kernel size k.
            dilation: spacing between kernel elements.
            norm_type: feature normalization type and arguments.
                Defaults to ``("batch", {"affine": True})``.
            acti_type: {``"relu"``, ``"prelu"``, ``"relu6"``}
                Non-linear activation using ReLU or PReLU. Defaults to ``"relu"``.
            channel_matching: {``"pad"``, ``"project"``}
                Specifies handling residual branch and conv branch channel mismatches. Defaults to ``"pad"``.

                - ``"pad"``: with zero padding.
                - ``"project"``: with a trainable conv with kernel size one.

        Raises:
            ValueError: When ``channel_matching=pad`` and ``in_channels > out_channels``. Incompatible values.

        """
        super(HighResBlock, self).__init__()
        self.chn_pad = ChannelPad(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, mode=channel_matching
        )

        layers = nn.ModuleList()
        _in_chns, _out_chns = in_channels, out_channels

        for kernel_size in kernels:
            layers.append(
                ADN(ordering="NA", in_channels=_in_chns, act=acti_type, norm=norm_type, norm_dim=spatial_dims)
            )
            layers.append(
                Convolution(
                    dimensions=spatial_dims,
                    in_channels=_in_chns,
                    out_channels=_out_chns,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            _in_chns = _out_chns

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv: torch.Tensor = self.layers(x)
        return x_conv + torch.as_tensor(self.chn_pad(x))


class HighResNet(nn.Module):
    """
    Reimplementation of highres3dnet based on
    Li et al., "On the compactness, efficiency, and representation of 3D
    convolutional networks: Brain parcellation as a pretext task", IPMI '17

    Adapted from:
    https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/network/highres3dnet.py
    https://github.com/fepegar/highresnet

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of input channels.
        out_channels: number of output channels.
        norm_type: feature normalization type and arguments.
            Defaults to ``("batch", {"affine": True})``.
        acti_type: activation type and arguments.
            Defaults to ``("relu", {"inplace": True})``.
        dropout_prob: probability of the feature map to be zeroed
            (only applies to the penultimate conv layer).
        layer_params: specifying key parameters of each layer/block.
        channel_matching: {``"pad"``, ``"project"``}
            Specifies handling residual branch and conv branch channel mismatches. Defaults to ``"pad"``.

            - ``"pad"``: with zero padding.
            - ``"project"``: with a trainable conv with kernel size one.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        norm_type: Union[str, tuple] = ("batch", {"affine": True}),
        acti_type: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout_prob: Optional[Union[Tuple, str, float]] = 0.0,
        layer_params: Sequence[Dict] = DEFAULT_LAYER_PARAMS_3D,
        channel_matching: Union[ChannelMatching, str] = ChannelMatching.PAD,
    ) -> None:

        super(HighResNet, self).__init__()
        blocks = nn.ModuleList()

        # initial conv layer
        params = layer_params[0]
        _in_chns, _out_chns = in_channels, params["n_features"]
        blocks.append(
            Convolution(
                dimensions=spatial_dims,
                in_channels=_in_chns,
                out_channels=_out_chns,
                kernel_size=params["kernel_size"],
                adn_ordering="NA",
                act=acti_type,
                norm=norm_type,
            )
        )

        # residual blocks
        for (idx, params) in enumerate(layer_params[1:-2]):  # res blocks except the 1st and last two conv layers.
            _in_chns, _out_chns = _out_chns, params["n_features"]
            _dilation = 2 ** idx
            for _ in range(params["repeat"]):
                blocks.append(
                    HighResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=_in_chns,
                        out_channels=_out_chns,
                        kernels=params["kernels"],
                        dilation=_dilation,
                        norm_type=norm_type,
                        acti_type=acti_type,
                        channel_matching=channel_matching,
                    )
                )
                _in_chns = _out_chns

        # final conv layers
        params = layer_params[-2]
        _in_chns, _out_chns = _out_chns, params["n_features"]
        blocks.append(
            Convolution(
                dimensions=spatial_dims,
                in_channels=_in_chns,
                out_channels=_out_chns,
                kernel_size=params["kernel_size"],
                adn_ordering="NAD",
                act=acti_type,
                norm=norm_type,
                dropout=dropout_prob,
            )
        )

        params = layer_params[-1]
        _in_chns = _out_chns
        blocks.append(
            Convolution(
                dimensions=spatial_dims,
                in_channels=_in_chns,
                out_channels=out_channels,
                kernel_size=params["kernel_size"],
                adn_ordering="NAD",
                act=acti_type,
                norm=norm_type,
                dropout=dropout_prob,
            )
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.blocks(x))
