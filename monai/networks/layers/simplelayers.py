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

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.convutils import gaussian_1d, same_padding
from monai.utils import ensure_tuple_rep

__all__ = ["SkipConnection", "Flatten", "GaussianFilter"]


class SkipConnection(nn.Module):
    """
    Concats the forward pass input with the result from the given submodule.
    """

    def __init__(self, submodule, cat_dim=1):
        super().__init__()
        self.submodule = submodule
        self.cat_dim = cat_dim

    def forward(self, x):
        return torch.cat([x, self.submodule(x)], self.cat_dim)


class Flatten(nn.Module):
    """
    Flattens the given input in the forward pass to be [B,-1] in shape.
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    """
    Reshapes input tensors to the given shape (minus batch dimension), retaining original batch size.
    """

    def __init__(self, *shape):
        """
        Given a shape list/tuple `shape` of integers (s0, s1, ... , sn), this layer will reshape input tensors of
        shape (batch, s0 * s1 * ... * sn) to shape (batch, s0, s1, ... , sn).

        Args:
            shape: list/tuple of integer shape dimensions
        """
        super().__init__()
        self.shape = (1,) + tuple(shape)

    def forward(self, x):
        shape = list(self.shape)
        shape[0] = x.shape[0]  # done this way for Torchscript
        return x.reshape(shape)


class GaussianFilter(nn.Module):
    def __init__(self, spatial_dims: int, sigma, truncated: float = 4.0):
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma (float or sequence of floats): std.
            truncated: spreads how many stds.
        """
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        _sigma = ensure_tuple_rep(sigma, self.spatial_dims)
        self.kernel = [
            torch.nn.Parameter(torch.as_tensor(gaussian_1d(s, truncated), dtype=torch.float), False) for s in _sigma
        ]
        self.padding = [same_padding(k.size()[0]) for k in self.kernel]
        self.conv_n = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
        for idx, param in enumerate(self.kernel):
            self.register_parameter(f"kernel_{idx}", param)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): in shape [Batch, chns, H, W, D].

        Raises:
            TypeError: x must be a Tensor, got {type(x).__name__}.

        """
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a Tensor, got {type(x).__name__}.")
        chns = x.shape[1]
        sp_dim = self.spatial_dims
        x = x.clone()  # no inplace change of x

        def _conv(input_, d):
            if d < 0:
                return input_
            s = [1] * (sp_dim + 2)
            s[d + 2] = -1
            kernel = self.kernel[d].reshape(s)
            kernel = kernel.repeat([chns, 1] + [1] * sp_dim)
            padding = [0] * sp_dim
            padding[d] = self.padding[d]
            return self.conv_n(input=_conv(input_, d - 1), weight=kernel, padding=padding, groups=chns)

        return _conv(x, sp_dim - 1)
