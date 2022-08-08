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

import torch.nn as nn
from torch import Tensor

from monai.apps.reconstruction.complex_utils import complex_abs
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.apps.reconstruction.networks.blocks.varnetblock import VarNetBlock
from monai.networks.blocks.fft_utils_t import ifftn_centered_t


class VariationalNetworkModel(nn.Module):
    """
    The end-to-end variational network (or simply e2e-VarNet) based on Sriram et. al., "End-to-end variational
    networks for accelerated MRI reconstruction".
    It comprises several cascades each consisting of refinement and data consistency steps. The network takes in
    the under-sampled kspace and estimates the ground-truth reconstruction.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        coil_sensitivity_model: A convolutional model for learning coil sensitivity maps. An example is
            :py:class:`monai.apps.reconstruction.networks.nets.coil_sensitivity_model.CoilSensitivityModel`.
        refinement_model: A convolutional network used in the refinement step of e2e-VarNet. An example
            is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet.ComplexUnet`.
        num_cascades: Number of cascades. Each cascade is a
            :py:class:`monai.apps.reconstruction.networks.blocks.varnetblock.VarNetBlock` which consists of
            refinement and data consistency steps.
        spatial_dims: number of spatial dimensions.
    """

    def __init__(
        self,
        coil_sensitivity_model: nn.Module,
        refinement_model: nn.Module,
        num_cascades: int = 12,
        spatial_dims: int = 2,
    ):
        super().__init__()
        self.coil_sensitivity_model = coil_sensitivity_model
        self.cascades = nn.ModuleList([VarNetBlock(refinement_model) for i in range(num_cascades)])
        self.spatial_dims = spatial_dims

    def forward(self, masked_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            masked_kspace: The under-sampled kspace. It's a 2D kspace (B,C,H,W,2)
                with the last dimension being 2 (for real/imaginary parts) and C denoting the
                coil dimension. 3D data will have the shape (B,C,H,W,D,2).
            mask: The under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.

        Returns:
            The reconstructed image which is the root sum of squares (rss) of the absolute value
                of the inverse fourier of the predicted kspace (note that rss combines coil images into one image).
        """
        sensitivity_maps = self.coil_sensitivity_model(masked_kspace, mask)  # shape is simlar to masked_kspace
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sensitivity_maps)

        output_image = root_sum_of_squares(
            complex_abs(ifftn_centered_t(kspace_pred, spatial_dims=self.spatial_dims)),
            spatial_dim=1,  # 1 is for C which is the coil dimension
        )  # shape is (B,H,W) for 2D and (B,H,W,D) for 3D data.
        return output_image  # type: ignore
