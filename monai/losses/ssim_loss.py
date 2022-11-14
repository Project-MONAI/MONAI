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

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from monai.metrics.regression import SSIMMetric

class SSIMLoss(_Loss):
    """
    Build a Pytorch version of the SSIM loss function based on the original formula of SSIM

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/ssim_loss_mixin.py

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, spatial_dims: int = 2):
        """
        Args:
            win_size: gaussian weighting window size
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            spatial_dims: if 2, input shape is expected to be (B,C,H,W). if 3, it is expected to be (B,C,H,W,D)
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.spatial_dims = spatial_dims
        

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D and pseudo-3D data,
                and (B,C,W,H,D) for 3D data,
            y: second sample (e.g., the reconstructed image). It has similar shape as x.
            data_range: dynamic range of the data

        Returns:
            1-ssim_value (recall this is meant to be a loss function)

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
        ssim_value = torch.empty((1), dtype=torch.float)
        if x.shape[0] == 1:
            ssim_value: torch.Tensor = SSIMMetric(data_range, self.win_size, self.k1, self.k2, self.spatial_dims)(
                x, y, data_range
            )
        elif x.shape[0] > 1:

            for i in range(x.shape[0]):
                ssim_val: torch.Tensor = SSIMMetric(data_range, self.win_size, self.k1, self.k2, self.spatial_dims)(
                    x[i : i + 1], y[i : i + 1],  data_range
                )
                if i == 0:
                    ssim_value = ssim_val
                else:
                    ssim_value = torch.cat((ssim_value.view(1), ssim_val.view(1)), dim=0)

        else:
            raise ValueError("Batch size is not nonnegative integer value")
        # 1- dimensional tensor is only allowed
        ssim_value = ssim_value.view(-1, 1)
        loss: torch.Tensor = 1 - ssim_value.mean()
        return loss
