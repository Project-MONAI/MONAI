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

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.convutils import same_padding
from monai.networks.layers.factories import Conv, Dropout, Norm
from monai.utils import Activation, ChannelMatching, Normalisation

SUPPORTED_NORM = {
    Normalisation.BATCH: lambda spatial_dims: Norm[Norm.BATCH, spatial_dims],
    Normalisation.INSTANCE: lambda spatial_dims: Norm[Norm.INSTANCE, spatial_dims],
}
SUPPORTED_ACTI = {Activation.RELU: nn.ReLU, Activation.PRELU: nn.PReLU, Activation.RELU6: nn.ReLU6}
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


class ConvNormActi(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: Optional[Union[Normalisation, str]] = None,
        acti_type: Optional[Union[Activation, str]] = None,
        dropout_prob: Optional[float] = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: size of the convolving kernel.
            norm_type: {``"batch"``, ``"instance"``}
                Feature normalisation with batchnorm or instancenorm. Defaults to ``"batch"``.
            acti_type: {``"relu"``, ``"prelu"``, ``"relu6"``}
                Non-linear activation using ReLU or PReLU. Defaults to ``"relu"``.
            dropout_prob: probability of the feature map to be zeroed
                (only applies to the penultimate conv layer).
        """

        super(ConvNormActi, self).__init__()

        layers = nn.ModuleList()

        conv_type = Conv[Conv.CONV, spatial_dims]
        padding_size = same_padding(kernel_size)
        conv = conv_type(in_channels, out_channels, kernel_size, padding=padding_size)
        layers.append(conv)

        if norm_type is not None:
            norm_type = Normalisation(norm_type)
            layers.append(SUPPORTED_NORM[norm_type](spatial_dims)(out_channels))
        if acti_type is not None:
            acti_type = Activation(acti_type)
            layers.append(SUPPORTED_ACTI[acti_type](inplace=True))
        if dropout_prob is not None:
            dropout_type = Dropout[Dropout.DROPOUT, spatial_dims]
            layers.append(dropout_type(p=dropout_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.layers(x))


class ChannelPad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = tuple(pad)

    def forward(self, x):
        return F.pad(x, self.pad)


class HighResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernels: Sequence[int] = (3, 3),
        dilation: Union[Sequence[int], int] = 1,
        norm_type: Union[Normalisation, str] = Normalisation.INSTANCE,
        acti_type: Union[Activation, str] = Activation.RELU,
        channel_matching: Union[ChannelMatching, str] = ChannelMatching.PAD,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernels: each integer k in `kernels` corresponds to a convolution layer with kernel size k.
            dilation: spacing between kernel elements.
            norm_type: {``"batch"``, ``"instance"``}
                Feature normalisation with batchnorm or instancenorm. Defaults to ``"instance"``.
            acti_type: {``"relu"``, ``"prelu"``, ``"relu6"``}
                Non-linear activation using ReLU or PReLU. Defaults to ``"relu"``.
            channel_matching: {``"pad"``, ``"project"``}
                Specifies handling residual branch and conv branch channel mismatches. Defaults to ``"pad"``.

                - ``"pad"``: with zero padding.
                - ``"project"``: with a trainable conv with kernel size.

        Raises:
            ValueError: When ``channel_matching=pad`` and ``in_channels > out_channels``. Incompatible values.

        """
        super(HighResBlock, self).__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Normalisation(norm_type)
        acti_type = Activation(acti_type)

        self.project = None
        self.pad = None

        if in_channels != out_channels:
            channel_matching = ChannelMatching(channel_matching)

            if channel_matching == ChannelMatching.PROJECT:
                self.project = conv_type(in_channels, out_channels, kernel_size=1)

            if channel_matching == ChannelMatching.PAD:
                if in_channels > out_channels:
                    raise ValueError('Incompatible values: channel_matching="pad" and in_channels > out_channels.')
                pad_1 = (out_channels - in_channels) // 2
                pad_2 = out_channels - in_channels - pad_1
                pad = [0, 0] * spatial_dims + [pad_1, pad_2] + [0, 0]
                self.pad = ChannelPad(pad)

        layers = nn.ModuleList()
        _in_chns, _out_chns = in_channels, out_channels

        for kernel_size in kernels:
            layers.append(SUPPORTED_NORM[norm_type](spatial_dims)(_in_chns))
            layers.append(SUPPORTED_ACTI[acti_type](inplace=True))
            layers.append(
                conv_type(
                    _in_chns, _out_chns, kernel_size, padding=same_padding(kernel_size, dilation), dilation=dilation
                )
            )
            _in_chns = _out_chns

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv: torch.Tensor = self.layers(x)

        if self.project is not None:
            return x_conv + torch.as_tensor(self.project(x))  # as_tensor used to get around mypy typing bug

        if self.pad is not None:
            return x_conv + torch.as_tensor(self.pad(x))

        return x_conv + x


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
        norm_type: {``"batch"``, ``"instance"``}
            Feature normalisation with batchnorm or instancenorm. Defaults to ``"batch"``.
        acti_type: {``"relu"``, ``"prelu"``, ``"relu6"``}
            Non-linear activation using ReLU or PReLU. Defaults to ``"relu"``.
        dropout_prob: probability of the feature map to be zeroed
            (only applies to the penultimate conv layer).
        layer_params: specifying key parameters of each layer/block.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        norm_type: Union[Normalisation, str] = Normalisation.BATCH,
        acti_type: Union[Activation, str] = Activation.RELU,
        dropout_prob: Optional[float] = None,
        layer_params: Sequence[Dict] = DEFAULT_LAYER_PARAMS_3D,
    ) -> None:

        super(HighResNet, self).__init__()
        blocks = nn.ModuleList()

        # initial conv layer
        params = layer_params[0]
        _in_chns, _out_chns = in_channels, params["n_features"]
        blocks.append(
            ConvNormActi(
                spatial_dims,
                _in_chns,
                _out_chns,
                kernel_size=params["kernel_size"],
                norm_type=norm_type,
                acti_type=acti_type,
                dropout_prob=None,
            )
        )

        # residual blocks
        for (idx, params) in enumerate(layer_params[1:-2]):  # res blocks except the 1st and last two conv layers.
            _in_chns, _out_chns = _out_chns, params["n_features"]
            _dilation = 2 ** idx
            for _ in range(params["repeat"]):
                blocks.append(
                    HighResBlock(
                        spatial_dims,
                        _in_chns,
                        _out_chns,
                        params["kernels"],
                        dilation=_dilation,
                        norm_type=norm_type,
                        acti_type=acti_type,
                    )
                )
                _in_chns = _out_chns

        # final conv layers
        params = layer_params[-2]
        _in_chns, _out_chns = _out_chns, params["n_features"]
        blocks.append(
            ConvNormActi(
                spatial_dims,
                _in_chns,
                _out_chns,
                kernel_size=params["kernel_size"],
                norm_type=norm_type,
                acti_type=acti_type,
                dropout_prob=dropout_prob,
            )
        )

        params = layer_params[-1]
        _in_chns = _out_chns
        blocks.append(
            ConvNormActi(
                spatial_dims,
                _in_chns,
                out_channels,
                kernel_size=params["kernel_size"],
                norm_type=norm_type,
                acti_type=None,
                dropout_prob=None,
            )
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.blocks(x))
