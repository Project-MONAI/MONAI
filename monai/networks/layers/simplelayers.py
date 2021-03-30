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

import math
from typing import List, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from monai.networks.layers.convutils import gaussian_1d
from monai.networks.layers.factories import Conv
from monai.utils import (
    PT_BEFORE_1_7,
    ChannelMatching,
    InvalidPyTorchVersionError,
    SkipMode,
    ensure_tuple_rep,
    optional_import,
)

_C, _ = optional_import("monai._C")
if not PT_BEFORE_1_7:
    fft, _ = optional_import("torch.fft")

__all__ = [
    "SkipConnection",
    "Flatten",
    "GaussianFilter",
    "LLTM",
    "Reshape",
    "separable_filtering",
    "SavitzkyGolayFilter",
    "HilbertTransform",
    "ChannelPad",
]


class ChannelPad(nn.Module):
    """
    Expand the input tensor's channel dimension from length `in_channels` to `out_channels`,
    by padding or a projection.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        mode: Union[ChannelMatching, str] = ChannelMatching.PAD,
    ):
        """

        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of input channels.
            out_channels: number of output channels.
            mode: {``"pad"``, ``"project"``}
                Specifies handling residual branch and conv branch channel mismatches. Defaults to ``"pad"``.

                - ``"pad"``: with zero padding.
                - ``"project"``: with a trainable conv with kernel size one.
        """
        super().__init__()
        self.project = None
        self.pad = None
        if in_channels == out_channels:
            return
        mode = ChannelMatching(mode)
        if mode == ChannelMatching.PROJECT:
            conv_type = Conv[Conv.CONV, spatial_dims]
            self.project = conv_type(in_channels, out_channels, kernel_size=1)
            return
        if mode == ChannelMatching.PAD:
            if in_channels > out_channels:
                raise ValueError('Incompatible values: channel_matching="pad" and in_channels > out_channels.')
            pad_1 = (out_channels - in_channels) // 2
            pad_2 = out_channels - in_channels - pad_1
            pad = [0, 0] * spatial_dims + [pad_1, pad_2] + [0, 0]
            self.pad = tuple(pad)
            return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            return torch.as_tensor(self.project(x))  # as_tensor used to get around mypy typing bug
        if self.pad is not None:
            return F.pad(x, self.pad)
        return x


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


def _separable_filtering_conv(
    input_: torch.Tensor,
    kernels: List[torch.Tensor],
    pad_mode: str,
    d: int,
    spatial_dims: int,
    paddings: List[int],
    num_channels: int,
) -> torch.Tensor:

    if d < 0:
        return input_

    s = [1] * len(input_.shape)
    s[d + 2] = -1
    _kernel = kernels[d].reshape(s)

    # if filter kernel is unity, don't convolve
    if _kernel.numel() == 1 and _kernel[0] == 1:
        return _separable_filtering_conv(input_, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels)

    _kernel = _kernel.repeat([num_channels, 1] + [1] * spatial_dims)
    _padding = [0] * spatial_dims
    _padding[d] = paddings[d]
    conv_type = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]

    # translate padding for input to torch.nn.functional.pad
    _reversed_padding_repeated_twice: List[List[int]] = [[p, p] for p in reversed(_padding)]
    _sum_reversed_padding_repeated_twice: List[int] = sum(_reversed_padding_repeated_twice, [])
    padded_input = F.pad(input_, _sum_reversed_padding_repeated_twice, mode=pad_mode)

    return conv_type(
        input=_separable_filtering_conv(padded_input, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels),
        weight=_kernel,
        groups=num_channels,
    )


def separable_filtering(x: torch.Tensor, kernels: List[torch.Tensor], mode: str = "zeros") -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.

    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all dimension), or `spatial_dims` number of kernels.
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``. Modes other than ``'zeros'`` require PyTorch version >= 1.5.1. See
            torch.nn.Conv1d() for more information.

    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    _kernels = [s.float() for s in kernels]
    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode

    return _separable_filtering_conv(x, kernels, pad_mode, spatial_dims - 1, spatial_dims, _paddings, n_chs)


class SavitzkyGolayFilter(nn.Module):
    """
    Convolve a Tensor along a particular axis with a Savitzky-Golay kernel.

    Args:
        window_length: Length of the filter window, must be a positive odd integer.
        order: Order of the polynomial to fit to each window, must be less than ``window_length``.
        axis (optional): Axis along which to apply the filter kernel. Default 2 (first spatial dimension).
        mode (string, optional): padding mode passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'`` or
        ``'circular'``. Default: ``'zeros'``. See torch.nn.Conv1d() for more information.
    """

    def __init__(self, window_length: int, order: int, axis: int = 2, mode: str = "zeros"):

        super().__init__()
        if order >= window_length:
            raise ValueError("order must be less than window_length.")

        self.axis = axis
        self.mode = mode
        self.coeffs = self._make_coeffs(window_length, order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to filter. Must be real, in shape ``[Batch, chns, spatial1, spatial2, ...]`` and
                have a device type of ``'cpu'``.
        Returns:
            torch.Tensor: ``x`` filtered by Savitzky-Golay kernel with window length ``self.window_length`` using
            polynomials of order ``self.order``, along axis specified in ``self.axis``.
        """

        # Make input a real tensor on the CPU
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError("Invalid axis for shape of x.")

        # Create list of filter kernels (1 per spatial dimension). The kernel for self.axis will be the savgol coeffs,
        # while the other kernels will be set to [1].
        n_spatial_dims = len(x.shape) - 2
        spatial_processing_axis = self.axis - 2
        new_dims_before = spatial_processing_axis
        new_dims_after = n_spatial_dims - spatial_processing_axis - 1
        kernel_list = [self.coeffs.to(device=x.device, dtype=x.dtype)]
        for _ in range(new_dims_before):
            kernel_list.insert(0, torch.ones(1, device=x.device, dtype=x.dtype))
        for _ in range(new_dims_after):
            kernel_list.append(torch.ones(1, device=x.device, dtype=x.dtype))

        return separable_filtering(x, kernel_list, mode=self.mode)

    @staticmethod
    def _make_coeffs(window_length, order):

        half_length, rem = divmod(window_length, 2)
        if rem == 0:
            raise ValueError("window_length must be odd.")

        idx = torch.arange(window_length - half_length - 1, -half_length - 1, -1, dtype=torch.float, device="cpu")
        a = idx ** torch.arange(order + 1, dtype=torch.float, device="cpu").reshape(-1, 1)
        y = torch.zeros(order + 1, dtype=torch.float, device="cpu")
        y[0] = 1.0
        return torch.lstsq(y, a).solution.squeeze()


class HilbertTransform(nn.Module):
    """
    Determine the analytical signal of a Tensor along a particular axis.
    Requires PyTorch 1.7.0+ and the PyTorch FFT module (which is not included in NVIDIA PyTorch Release 20.10).

    Args:
        axis: Axis along which to apply Hilbert transform. Default 2 (first spatial dimension).
        N: Number of Fourier components (i.e. FFT size). Default: ``x.shape[axis]``.
    """

    def __init__(self, axis: int = 2, n: Union[int, None] = None) -> None:

        if PT_BEFORE_1_7:
            raise InvalidPyTorchVersionError("1.7.0", self.__class__.__name__)

        super().__init__()
        self.axis = axis
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to transform. Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
        Returns:
            torch.Tensor: Analytical signal of ``x``, transformed along axis specified in ``self.axis`` using
            FFT of size ``self.N``. The absolute value of ``x_ht`` relates to the envelope of ``x`` along axis ``self.axis``.
        """

        # Make input a real tensor
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError("Invalid axis for shape of x.")

        n = x.shape[self.axis] if self.n is None else self.n
        if n <= 0:
            raise ValueError("N must be positive.")
        x = torch.as_tensor(x, dtype=torch.complex64)
        # Create frequency axis
        f = torch.cat(
            [
                torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
                torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
            ]
        )
        xf = fft.fft(x, n=n, dim=self.axis)
        # Create step function
        u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
        new_dims_before = self.axis
        new_dims_after = len(xf.shape) - self.axis - 1
        for _ in range(new_dims_before):
            u.unsqueeze_(0)
        for _ in range(new_dims_after):
            u.unsqueeze_(-1)

        ht = fft.ifft(xf * 2 * u, dim=self.axis)

        # Apply transform
        return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)


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
                torch.as_tensor(s, dtype=torch.float, device=s.device if isinstance(s, torch.Tensor) else None),
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
