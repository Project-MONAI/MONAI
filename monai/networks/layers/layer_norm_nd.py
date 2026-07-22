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


class LayerNormNd(nn.Module):
    """
    Layer normalization over the channel dimension of a channels-first tensor.

    `torch.nn.LayerNorm` normalizes over the trailing dimensions, so it expects a channels-last layout
    such as ``(batch, *spatial, channel)``. Convolutional feature maps in MONAI are channels-first,
    ``(batch, channel, *spatial)``, and this module normalizes those over the channel dimension only,
    for any number of spatial dimensions.

    Args:
        num_channels: number of channels of the input, i.e. the size of dimension 1.
        spatial_dims: number of spatial dimensions of the input image.
        eps: value added to the denominator for numerical stability.
    """

    def __init__(self, num_channels: int, spatial_dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        # broadcast the affine parameters against (batch, channel, *spatial); precomputed so that the
        # module is scriptable without inspecting the rank of the input at runtime.
        self.param_shape = [1, num_channels] + [1] * spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(self.param_shape) * x + self.bias.view(self.param_shape)
