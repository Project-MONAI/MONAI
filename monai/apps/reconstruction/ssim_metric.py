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

from monai.metrics.regression import RegressionMetric


class SSIMMetric(RegressionMetric):
    r"""
    Build a Pytorch version of the SSIM metric based on the original formula of SSIM

    .. math::
        \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/ssim_loss_mixin.py

    Args:
        data_range: dynamic range of the data
        win_size: gaussian weighting window size
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
    """

    def __init__(self, data_range: torch.Tensor, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self._synced_tensors = [torch.ones(1, 1, win_size, win_size) / win_size**2]  # buffered weights
        self.cov_norm = (win_size**2) / (win_size**2 - 1)

    def _compute_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y and x are of shape (B,C,H,W) where B is batch_size, C is number of channels,
            and (H,W) are height and width. C can also serve as the slice dimension to pass volumes.

        Args:
            x: first sample (e.g., the reference image).
            y: second sample (e.g., the reconstructed image)
            data_range: dynamic range of the data

        Returns:
            ssim_value

        Example:
            .. code-block:: python

                import torch
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(SSIMMetric()._compute_metric(x,y,data_range))
        """
        data_range = self.data_range[:, None, None, None]
        c1 = (self.k1 * data_range) ** 2  # stability constant for luminance
        c2 = (self.k2 * data_range) ** 2  # stability constant for contrast
        ux = F.conv2d(x, self.get_buffer())  # mu_x
        uy = F.conv2d(y, self.get_buffer())  # mu_y
        uxx = F.conv2d(x * x, self.get_buffer())  # mu_x^2
        uyy = F.conv2d(y * y, self.get_buffer())  # mu_y^2
        uxy = F.conv2d(x * y, self.get_buffer())  # mu_xy
        vx = self.cov_norm * (uxx - ux * ux)  # sigma_x
        vy = self.cov_norm * (uyy - uy * uy)  # sigma_y
        vxy = self.cov_norm * (uxy - ux * uy)  # sigma_xy

        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denom = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        ssim_value: torch.Tensor = (numerator / denom).mean()
        return ssim_value
