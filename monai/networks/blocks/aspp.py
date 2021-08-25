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

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import same_padding
from monai.networks.layers.factories import Conv


class SimpleASPP(nn.Module):
    """
    A simplified version of the atrous spatial pyramid pooling (ASPP) module.

    Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
    https://arxiv.org/abs/1802.02611

    Wang et al., A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions
    from CT Images. https://ieeexplore.ieee.org/document/9109297
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        conv_out_channels: int,
        kernel_sizes: Sequence[int] = (1, 3, 3, 3),
        dilations: Sequence[int] = (1, 2, 4, 6),
        norm_type: Optional[Union[Tuple, str]] = "BATCH",
        acti_type: Optional[Union[Tuple, str]] = "LEAKYRELU",
        bias: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            conv_out_channels: number of output channels of each atrous conv.
                The final number of output channels is conv_out_channels * len(kernel_sizes).
            kernel_sizes: a sequence of four convolutional kernel sizes.
                Defaults to (1, 3, 3, 3) for four (dilated) convolutions.
            dilations: a sequence of four convolutional dilation parameters.
                Defaults to (1, 2, 4, 6) for four (dilated) convolutions.
            norm_type: final kernel-size-one convolution normalization type.
                Defaults to batch norm.
            acti_type: final kernel-size-one convolution activation type.
                Defaults to leaky ReLU.
            bias: whether to have a bias term in convolution blocks. Defaults to False.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.

        Raises:
            ValueError: When ``kernel_sizes`` length differs from ``dilations``.

        See also:

            :py:class:`monai.networks.layers.Act`
            :py:class:`monai.networks.layers.Conv`
            :py:class:`monai.networks.layers.Norm`

        """
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError(
                "kernel_sizes and dilations length must match, "
                f"got kernel_sizes={len(kernel_sizes)} dilations={len(dilations)}."
            )
        pads = tuple(same_padding(k, d) for k, d in zip(kernel_sizes, dilations))

        self.convs = nn.ModuleList()
        for k, d, p in zip(kernel_sizes, dilations, pads):
            _conv = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=conv_out_channels, kernel_size=k, dilation=d, padding=p
            )
            self.convs.append(_conv)

        out_channels = conv_out_channels * len(pads)  # final conv. output channels
        self.conv_k1 = Convolution(
            dimensions=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=acti_type,
            norm=norm_type,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
        """
        x_out = torch.cat([conv(x) for conv in self.convs], dim=1)
        x_out = self.conv_k1(x_out)
        return x_out
