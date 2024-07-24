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

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_norm_layer


class SPADE(nn.Module):
    """
    Spatially Adaptive Normalization (SPADE) block, allowing for normalization of activations conditioned on a
    semantic map. This block is used in SPADE-based image-to-image translation models, as described in
    Semantic Image Synthesis with Spatially-Adaptive Normalization (https://arxiv.org/abs/1903.07291).

    Args:
        label_nc: number of semantic labels
        norm_nc: number of output channels
        kernel_size: kernel size
        spatial_dims: number of spatial dimensions
        hidden_channels: number of channels in the intermediate gamma and beta layers
        norm: type of base normalisation used before applying the SPADE normalisation
        norm_params: parameters for the base normalisation
    """

    def __init__(
        self,
        label_nc: int,
        norm_nc: int,
        kernel_size: int = 3,
        spatial_dims: int = 2,
        hidden_channels: int = 64,
        norm: str | tuple = "INSTANCE",
        norm_params: dict | None = None,
    ) -> None:
        super().__init__()

        if norm_params is None:
            norm_params = {}
        if len(norm_params) != 0:
            norm = (norm, norm_params)
        self.param_free_norm = get_norm_layer(norm, spatial_dims=spatial_dims, channels=norm_nc)
        self.mlp_shared = Convolution(
            spatial_dims=spatial_dims,
            in_channels=label_nc,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            norm=None,
            act="LEAKYRELU",
        )
        self.mlp_gamma = Convolution(
            spatial_dims=spatial_dims,
            in_channels=hidden_channels,
            out_channels=norm_nc,
            kernel_size=kernel_size,
            act=None,
        )
        self.mlp_beta = Convolution(
            spatial_dims=spatial_dims,
            in_channels=hidden_channels,
            out_channels=norm_nc,
            kernel_size=kernel_size,
            act=None,
        )

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor with shape (B, C, [spatial-dimensions]) where C is the number of semantic channels.
            segmap: input segmentation map (B, C, [spatial-dimensions]) where C is the number of semantic channels.
            The map will be interpolated to the dimension of x internally.
        """

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x.contiguous())

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out: torch.Tensor = normalized * (1 + gamma) + beta
        return out
