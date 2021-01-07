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
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction, Union

conv_dict = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


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
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            in_channels: number of input channels
            ndim: number of spatial ndimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel size or kernel sigma for kernel_type=``"gaussian"``
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super(LocalNormalizedCrossCorrelationLoss, self).__init__(reduction=LossReduction(reduction).value)
        self.in_channels = in_channels
        self.ndim = ndim
        if self.ndim not in [1, 2, 3]:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")
        self.fn = conv_dict[self.ndim]
        self.kernel_size = kernel_size
        if kernel_type == "rectangular":
            self.kernel, self.kernel_vol, self.padding = self.make_rectangular_kernel()
        elif kernel_type == "triangular":
            self.kernel, self.kernel_vol, self.padding = self.make_triangular_kernel()
        elif kernel_type == "gaussian":
            self.kernel, self.kernel_vol, self.padding = self.make_gaussian_kernel()
        else:
            raise ValueError(
                f'Unsupported kernel_type: {kernel_type}, available options are ["rectangular", "triangular", "gaussian"].'
            )
        self.smooth_dr = float(smooth_dr)

    def make_rectangular_kernel(self):
        shape = [1, self.in_channels] + [self.kernel_size] * self.ndim
        return torch.ones(shape, dtype=torch.float), self.kernel_size ** self.ndim, int((self.kernel_size - 1) / 2)

    def make_triangular_kernel(self):
        fsize = int((self.kernel_size + 1) // 2)
        f1 = torch.ones([1, 1] + [fsize] * self.ndim, dtype=torch.float).div(fsize)  # (1, 1, D, H, W)
        f1 = F.pad(f1, [(fsize - 1) // 2, (fsize - 1) // 2] * self.ndim)
        f2 = torch.ones([self.in_channels, 1] + [fsize] * self.ndim, dtype=torch.float).div(fsize)
        # (in_channels, 1, D, H, W)
        # (1, 1, D, H, W) -> (1, in_channels, D, H, W)
        padding_needed = max(fsize - 1, 0)
        padding = [padding_needed // 2, padding_needed - padding_needed // 2] * self.ndim
        f1 = F.pad(f1, padding)
        kernel = self.fn(f1, f2)

        return kernel, torch.sum(kernel ** 2), int((fsize - 1) / 2.0)

    def make_gaussian_kernel(self):
        mean = (self.kernel_size - 1) / 2.0
        sigma = self.kernel_size / 3.0

        grid_ndim = torch.arange(0, self.kernel_size)
        grid_ndim_ch = torch.arange(0, self.in_channels)

        if self.ndim == 1:
            grid = torch.meshgrid(grid_ndim_ch, grid_ndim)
        elif self.ndim == 2:
            grid = torch.meshgrid(grid_ndim_ch, grid_ndim, grid_ndim)
        elif self.ndim == 3:
            grid = torch.meshgrid(grid_ndim_ch, grid_ndim, grid_ndim, grid_ndim)
        else:
            raise ValueError

        grid = torch.stack(grid, dim=-1).to(dtype=torch.float)
        kernel = torch.exp(-torch.sum(torch.square(grid - mean), dim=-1) / (2 * sigma ** 2)).unsqueeze(
            0
        )  # (1, in_channel, kernel_size, kernel_size, kernel_size)
        return kernel, torch.sum(kernel ** 2), int((self.kernel_size - 1) / 2.0)

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
        t_sum = self.fn(target, weight=self.kernel, padding=self.padding)
        p_sum = self.fn(input, weight=self.kernel, padding=self.padding)
        t2_sum = self.fn(t2, weight=self.kernel, padding=self.padding)
        p2_sum = self.fn(p2, weight=self.kernel, padding=self.padding)
        tp_sum = self.fn(tp, weight=self.kernel, padding=self.padding)

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
        ncc = (cross * cross + self.smooth_dr) / (t_var * p_var + self.smooth_dr)  # shape = (batch, 1, D, H, W)

        if self.reduction == LossReduction.SUM.value:
            return -torch.sum(ncc).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return ncc.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(ncc).neg()  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
