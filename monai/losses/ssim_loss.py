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

from collections.abc import Sequence

import torch
from torch.nn.modules.loss import _Loss

from monai.metrics.regression import KernelType, SSIMMetric
from monai.utils import LossReduction, ensure_tuple_rep


class SSIMLoss(_Loss):
    """
    Compute the loss function based on the Structural Similarity Index Measure (SSIM) Metric.

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        win_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions of the input images.
            data_range: value range of input images. (usually 1.0 or 255)
            kernel_type: type of kernel, can be "gaussian" or "uniform".
            win_size: window size of kernel
            kernel_sigma: standard deviation for Gaussian kernel.
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.spatial_dims = spatial_dims
        self._data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(win_size, Sequence):
            win_size = ensure_tuple_rep(win_size, spatial_dims)
        self.kernel_size = win_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2

        self.ssim_metric = SSIMMetric(
            spatial_dims=self.spatial_dims,
            data_range=self._data_range,
            kernel_type=self.kernel_type,
            win_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

    @property
    def data_range(self) -> float:
        return self._data_range

    @data_range.setter
    def data_range(self, value: float) -> None:
        self._data_range = value
        self.ssim_metric.data_range = value

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: batch of predicted images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
            target: batch of target images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])

        Returns:
            1 minus the ssim index (recall this is meant to be a loss function)

        Example:
            .. code-block:: python

                import torch

                # 2D data
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                print(1-SSIMLoss(spatial_dims=2)(x,y))

                # pseudo-3D data
                x = torch.ones([1,5,10,10])/2  # 5 could represent number of slices
                y = torch.ones([1,5,10,10])/2
                print(1-SSIMLoss(spatial_dims=2)(x,y))

                # 3D data
                x = torch.ones([1,1,10,10,10])/2
                y = torch.ones([1,1,10,10,10])/2
                print(1-SSIMLoss(spatial_dims=3)(x,y))
        """
        ssim_value = self.ssim_metric._compute_tensor(input, target).view(-1, 1)
        loss: torch.Tensor = 1 - ssim_value

        if self.reduction == LossReduction.MEAN.value:
            loss = torch.mean(loss)  # the batch average
        elif self.reduction == LossReduction.SUM.value:
            loss = torch.sum(loss)  # sum over the batch

        return loss
