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


class SSIM(nn.Module):
    """
    Build a Pytorch version of the SSIM loss function based on the original formula of SSIM

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/ssim_loss_mixin.py
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        np = win_size**2
        self.cov_norm = np / (np - 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """
        y and x are of shape (B,C,H,W) where B is batch_size, C is number of channels, and (H,W) are height and width.

        Args:
            x: first sample (e.g., the reference image).
            y: second sample (e.g., the reconstructed image)
            data_range: dynamic range of the data

        Returns:
            1-ssim_score (recall this is meant to be a loss function)

        Example:
            .. code-block:: python

                import torch
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIM()(x,y,data_range))
        """
        data_range = data_range[:, None, None, None]
        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(x, convert_to_tensor(self.w))
        uy = F.conv2d(y, convert_to_tensor(self.w))
        uxx = F.conv2d(x * x, convert_to_tensor(self.w))
        uyy = F.conv2d(y * y, convert_to_tensor(self.w))
        uxy = F.conv2d(x * y, convert_to_tensor(self.w))
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        a1, a2, b1, b2 = (2 * ux * uy + c1, 2 * vxy + c2, ux**2 + uy**2 + c1, vx + vy + c2)
        d = b1 * b2
        s = (a1 * a2) / d
        loss: torch.Tensor = 1 - s.mean()
        return loss
