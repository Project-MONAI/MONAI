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

import torch

from monai.utils.module import optional_import

_C, _ = optional_import("monai._C")

__all__ = ["BilateralFilter", "PHLFilter"]


class BilateralFilter(torch.autograd.Function):
    """
    Blurs the input tensor spatially whilst preserving edges. Can run on 1D, 2D, or 3D,
    tensors (on top of Batch and Channel dimensions). Two implementations are provided,
    an exact solution and a much faster approximation which uses a permutohedral lattice.

    See:
        https://en.wikipedia.org/wiki/Bilateral_filter
        https://graphics.stanford.edu/papers/permutohedral/

    Args:
        input: input tensor.

        spatial sigma: the standard deviation of the spatial blur. Higher values can
            hurt performance when not using the approximate method (see fast approx).

        color sigma: the standard deviation of the color blur. Lower values preserve
            edges better whilst higher values tend to a simple gaussian spatial blur.

        fast approx: This flag chooses between two implementations. The approximate method may
            produce artifacts in some scenarios whereas the exact solution may be intolerably
            slow for high spatial standard deviations.

    Returns:
        output (torch.Tensor): output tensor.
    """

    @staticmethod
    def forward(ctx, input, spatial_sigma=5, color_sigma=0.5, fast_approx=True):
        ctx.ss = spatial_sigma
        ctx.cs = color_sigma
        ctx.fa = fast_approx
        output_data = _C.bilateral_filter(input, spatial_sigma, color_sigma, fast_approx)
        return output_data

    @staticmethod
    def backward(ctx, grad_output):
        spatial_sigma, color_sigma, fast_approx = ctx.ss, ctx.cs, ctx.fa
        grad_input = _C.bilateral_filter(grad_output, spatial_sigma, color_sigma, fast_approx)
        return grad_input, None, None, None


class PHLFilter(torch.autograd.Function):
    """
    Filters input based on arbitrary feature vectors. Uses a permutohedral
    lattice data structure to efficiently approximate n-dimensional gaussian
    filtering. Complexity is broadly independent of kernel size. Most applicable
    to higher filter dimensions and larger kernel sizes.

    See:
        https://graphics.stanford.edu/papers/permutohedral/

    Args:
        input: input tensor to be filtered.

        features: feature tensor used to filter the input.

        sigmas: the standard deviations of each feature in the filter.

    Returns:
        output (torch.Tensor): output tensor.
    """

    @staticmethod
    def forward(ctx, input, features, sigmas=None):

        scaled_features = features
        if sigmas is not None:
            for i in range(features.size(1)):
                scaled_features[:, i, ...] /= sigmas[i]

        ctx.save_for_backward(scaled_features)
        output_data = _C.phl_filter(input, scaled_features)
        return output_data

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("PHLFilter does not currently support Backpropagation")
        # scaled_features, = ctx.saved_variables
        # grad_input = _C.phl_filter(grad_output, scaled_features)
        # return grad_input
