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
from monai.utils import ensure_tuple_rep


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
        kernel_size: int | Sequence[int, ...] = 11,
        kernel_sigma: float | Sequence[float, ...] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions of the input images.
            data_range: value range of input images. (usually 1.0 or 255)
            kernel_type: type of kernel, can be "gaussian" or "uniform".
            kernel_size: size of kernel
            kernel_sigma: standard deviation for Gaussian kernel.
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(kernel_size, Sequence):
            kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        self.kernel_size = kernel_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2

        self.ssim_metric = SSIMMetric(
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: batch of predicted images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
            y: batch of target images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])

        Returns:
            1 minus the Structural Similarity Index Measure (recall this is meant to be a loss function)

        Example:
            .. code-block:: python

                import torch

                # 2D data
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=2)(x,y,data_range))

                # pseudo-3D data
                x = torch.ones([1,5,10,10])/2  # 5 could represent number of slices
                y = torch.ones([1,5,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=2)(x,y,data_range))

                # 3D data
                x = torch.ones([1,1,10,10,10])/2
                y = torch.ones([1,1,10,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=3)(x,y,data_range))
        """
        if x.shape[0] == 1:
            ssim_value: torch.Tensor = SSIMMetric(
                data_range, self.win_size, self.k1, self.k2, self.spatial_dims
            )._compute_tensor(x, y)
        elif x.shape[0] > 1:
            for i in range(x.shape[0]):
                ssim_val: torch.Tensor = SSIMMetric(
                    data_range, self.win_size, self.k1, self.k2, self.spatial_dims
                )._compute_tensor(x[i : i + 1], y[i : i + 1])
                if i == 0:
                    ssim_value = ssim_val
                else:
                    ssim_value = torch.cat((ssim_value.view(i), ssim_val.view(1)), dim=0)

        else:
            raise ValueError("Batch size is not nonnegative integer value")
        # 1- dimensional tensor is only allowed
        ssim_value = ssim_value.view(-1, 1)
        loss: torch.Tensor = 1 - ssim_value.mean()
        return loss
