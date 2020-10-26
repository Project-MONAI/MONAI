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
from monai.utils import SkipMode, ensure_tuple_rep, optional_import

_C, _ = optional_import("monai._C")

__all__ = ["SkipConnection", "Flatten", "GaussianFilter", "LLTM", "Reshape", "separable_filtering"]


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: Union[str, SkipMode] = "cat") -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = SkipMode(mode).value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


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


def separable_filtering(x: torch.Tensor, kernels: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.

    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all dimension), or `spatial_dims` number of kernels.

    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    _kernels = [
        torch.as_tensor(s, dtype=torch.float, device=s.device if torch.is_tensor(s) else None)
        for s in ensure_tuple_rep(kernels, spatial_dims)
    ]
    _paddings = [cast(int, (same_padding(k.shape[0]))) for k in _kernels]
    n_chns = x.shape[1]

    def _conv(input_: torch.Tensor, d: int) -> torch.Tensor:
        if d < 0:
            return input_
        s = [1] * len(input_.shape)
        s[d + 2] = -1
        _kernel = kernels[d].reshape(s)
        _kernel = _kernel.repeat([n_chns, 1] + [1] * spatial_dims)
        _padding = [0] * spatial_dims
        _padding[d] = _paddings[d]
        conv_type = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
        return conv_type(input=_conv(input_, d - 1), weight=_kernel, padding=_padding, groups=n_chns)

    return _conv(x, spatial_dims - 1)


class GaussianFilter(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor],
        truncated: float = 4.0,
        approx: str = "erf",
        requires_grad: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std. could be a single value, or `spatial_dims` number of values.
            truncated: spreads how many stds.
            approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

                - ``erf`` approximation interpolates the error function;
                - ``sampled`` uses a sampled Gaussian kernel;
                - ``scalespace`` corresponds to
                  https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
                  based on the modified Bessel functions.

            requires_grad: whether to store the gradients for sigma.
                if True, `sigma` will be the initial value of the parameters of this module
                (for example `parameters()` iterator could be used to get the parameters);
                otherwise this module will fix the kernels using `sigma` as the std.
        """
        super().__init__()
        self.sigma = [
            torch.nn.Parameter(
                torch.as_tensor(s, dtype=torch.float, device=s.device if torch.is_tensor(s) else None),
                requires_grad=requires_grad,
            )
            for s in ensure_tuple_rep(sigma, int(spatial_dims))
        ]
        self.truncated = truncated
        self.approx = approx
        for idx, param in enumerate(self.sigma):
            self.register_parameter(f"kernel_sigma_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].
        """
        _kernel = [gaussian_1d(s, truncated=self.truncated, approx=self.approx) for s in self.sigma]
        return separable_filtering(x=x, kernels=_kernel)


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
