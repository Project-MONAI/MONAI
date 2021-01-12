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
from typing import Union

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.networks.layers import gaussian_1d, separable_filtering
from monai.utils import LossReduction


def make_rectangular_kernel(kernel_size: int) -> torch.Tensor:
    return torch.ones(kernel_size)


def make_triangular_kernel(kernel_size: int) -> torch.Tensor:
    fsize = (kernel_size + 1) // 2
    if fsize % 2 == 0:
        fsize -= 1
    f = torch.ones((1, 1, fsize), dtype=torch.float).div(fsize)
    padding = (kernel_size - fsize) // 2 + fsize // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)


def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    sigma = torch.tensor(kernel_size / 3.0)
    kernel = gaussian_1d(sigma=sigma, truncated=kernel_size // 2, approx="sampled", normalize=False) * (
        2.5066282 * sigma
    )
    return kernel[:kernel_size]


kernel_dict = {
    "rectangular": make_rectangular_kernel,
    "triangular": make_triangular_kernel,
    "gaussian": make_gaussian_kernel,
}


class LocalNormalizedCrossCorrelationLoss(_Loss):
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.

    Adapted from:
        https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        in_channels: int,
        ndim: int = 3,
        kernel_size: int = 9,
        kernel_type: str = "rectangular",
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            in_channels: number of input channels
            ndim: number of spatial ndimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super(LocalNormalizedCrossCorrelationLoss, self).__init__(reduction=LossReduction(reduction).value)
        self.in_channels = in_channels

        self.ndim = ndim
        if self.ndim not in [1, 2, 3]:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        if kernel_type not in kernel_dict.keys():
            raise ValueError(
                f'Unsupported kernel_type: {kernel_type}, available options are ["rectangular", "triangular", "gaussian"].'
            )
        self.kernel = kernel_dict[kernel_type](self.kernel_size)
        self.kernel_vol = torch.sum(self.kernel) ** self.ndim
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        assert (
            input.shape[1] == self.in_channels
        ), f"expecting input with {self.in_channels} channels, got input of shape {input.shape}"
        assert (
            input.ndim - 2 == self.ndim
        ), f"expecting input with {self.ndim} spatial dimensions, got input of shape {input.shape}"
        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        t2, p2, tp = target ** 2, input ** 2, target * input

        # sum over kernel
        t_sum = separable_filtering(target, kernels=[self.kernel] * self.ndim).sum(1, keepdim=True)
        p_sum = separable_filtering(input, kernels=[self.kernel] * self.ndim).sum(1, keepdim=True)
        t2_sum = separable_filtering(t2, kernels=[self.kernel] * self.ndim).sum(1, keepdim=True)
        p2_sum = separable_filtering(p2, kernels=[self.kernel] * self.ndim).sum(1, keepdim=True)
        tp_sum = separable_filtering(tp, kernels=[self.kernel] * self.ndim).sum(1, keepdim=True)

        # average over kernel
        t_avg = t_sum / self.kernel_vol
        p_avg = p_sum / self.kernel_vol

        # normalized cross correlation between t and p
        # sum[(t - mean[t]) * (p - mean[p])] / std[t] / std[p]
        # denoted by num / denom
        # assume we sum over N values
        # num = sum[t * p - mean[t] * p - t * mean[p] + mean[t] * mean[p]]
        #     = sum[t*p] - sum[t] * sum[p] / N * 2 + sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * mean[p] = cross
        # the following is actually squared ncc
        cross = tp_sum - p_avg * t_sum
        t_var = t2_sum - t_avg * t_sum  # std[t] ** 2
        p_var = p2_sum - p_avg * p_sum  # std[p] ** 2
        ncc: torch.Tensor = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)
        # shape = (batch, 1, D, H, W)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(ncc).neg()  # sum over the batch and spatial ndims
        if self.reduction == LossReduction.NONE.value:
            return ncc.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(ncc).neg()  # average over the batch and spatial ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
