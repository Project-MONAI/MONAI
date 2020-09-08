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

import math
from typing import Sequence, Union, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from monai.networks.layers.convutils import gaussian_1d, same_padding
from monai.utils import ensure_tuple_rep, optional_import

_C, _ = optional_import("monai._C")

__all__ = ["SkipConnection", "Flatten", "GaussianFilter", "LLTM"]


class SkipConnection(nn.Module):
    """
    Concats the forward pass input with the result from the given submodule.
    """

    def __init__(self, submodule, cat_dim: int = 1) -> None:
        super().__init__()
        self.submodule = submodule
        self.cat_dim = cat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.submodule(x)], self.cat_dim)


class Flatten(nn.Module):
    """
    Flattens the given input in the forward pass to be [B,-1] in shape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    """
    Reshapes input tensors to the given shape (minus batch dimension), retaining original batch size.
    """

    def __init__(self, *shape: int) -> None:
        """
        Given a shape list/tuple `shape` of integers (s0, s1, ... , sn), this layer will reshape input tensors of
        shape (batch, s0 * s1 * ... * sn) to shape (batch, s0, s1, ... , sn).

        Args:
            shape: list/tuple of integer shape dimensions
        """
        super().__init__()
        self.shape = (1,) + tuple(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(self.shape)
        shape[0] = x.shape[0]  # done this way for Torchscript
        return x.reshape(shape)


class GaussianFilter(nn.Module):
    def __init__(self, spatial_dims: int, sigma: Union[Sequence[float], float], truncated: float = 4.0) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std.
            truncated: spreads how many stds.
        """
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        _sigma = ensure_tuple_rep(sigma, self.spatial_dims)
        self.kernel = [
            torch.nn.Parameter(torch.as_tensor(gaussian_1d(s, truncated), dtype=torch.float), False) for s in _sigma
        ]
        self.padding = [cast(int, (same_padding(k.size()[0]))) for k in self.kernel]
        self.conv_n = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
        for idx, param in enumerate(self.kernel):
            self.register_parameter(f"kernel_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].

        Raises:
            TypeError: When ``x`` is not a ``torch.Tensor``.

        """
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
        chns = x.shape[1]
        sp_dim = self.spatial_dims
        x = x.clone()  # no inplace change of x

        def _conv(input_: torch.Tensor, d: int) -> torch.Tensor:
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


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = _C.lltm_forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = _C.lltm_backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs[:5]

        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    """
    This recurrent unit is similar to an LSTM, but differs in that it lacks a forget
    gate and uses an Exponential Linear Unit (ELU) as its internal activation function.
    Because this unit never forgets, call it LLTM, or Long-Long-Term-Memory unit.
    It has both C++ and CUDA implementation, automatically switch according to the
    target device where put this module to.

    Args:
        input_features: size of input feature data
        state_size: size of the state of recurrent unit

    Referring to: https://pytorch.org/tutorials/advanced/cpp_extension.html
    """

    def __init__(self, input_features: int, state_size: int):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(torch.empty(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.empty(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
