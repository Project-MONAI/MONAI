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
from torch import nn

from monai.utils.type_conversion import convert_to_tensor


class SSIMLoss(nn.Module):
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
            spatial_dims: if 2, input shape is expected to be (B,C,W,H). if 3, it is expected to be (B,C,W,H,D)
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.spatial_dims = spatial_dims
        self.register_buffer(
            "w", torch.ones([1, 1] + [win_size for _ in range(spatial_dims)]) / win_size**spatial_dims
        )
        self.cov_norm = (win_size**2) / (win_size**2 - 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D data and (B,C,W,H,D) for 3D.
                A fastMRI sample should use the 2D format with C being the number of slices.
            y: second sample (e.g., the reconstructed image). It has similar shape as x.
            data_range: dynamic range of the data

        Returns:
            1-ssim_value (recall this is meant to be a loss function)

        Example:
            .. code-block:: python

                import torch
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=2)(x,y,data_range))
        """
        data_range = data_range[(None,) * (self.spatial_dims + 2)]
        # determine whether to work with 2D convolution or 3D
        conv = getattr(F, f"conv{self.spatial_dims}d")

        c1 = (self.k1 * data_range) ** 2  # stability constant for luminance
        c2 = (self.k2 * data_range) ** 2  # stability constant for contrast
        ux = conv(x, convert_to_tensor(self.w))  # mu_x
        uy = conv(y, convert_to_tensor(self.w))  # mu_y
        uxx = conv(x * x, convert_to_tensor(self.w))  # mu_x^2
        uyy = conv(y * y, convert_to_tensor(self.w))  # mu_y^2
        uxy = conv(x * y, convert_to_tensor(self.w))  # mu_xy
        vx = self.cov_norm * (uxx - ux * ux)  # sigma_x
        vy = self.cov_norm * (uyy - uy * uy)  # sigma_y
        vxy = self.cov_norm * (uxy - ux * uy)  # sigma_xy

        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denom = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        ssim_value = numerator / denom
        loss: torch.Tensor = 1 - ssim_value.mean()
        return loss
