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
import torch.nn as nn
from torch import Tensor

from monai.apps.reconstruction.complex_utils import complex_conj, complex_mul
from monai.networks.blocks.fft_utils_t import fftn_centered_t, ifftn_centered_t


class VarNetBlock(nn.Module):
    """
    A variational block based on Sriram et. al., "End-to-end variational networks for accelerated MRI reconstruction".
    It applies data consistency and refinement to the intermediate kspace and combines those results.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        model: the model used for refinement (typically a U-Net but can be any standard deep learning model)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.register_buffer("zeros", torch.zeros(1, 1, 1, 1, 1))

    def forward(self, current_kspace: Tensor, ref_kspace: Tensor, mask: Tensor, sens_maps: Tensor) -> Tensor:
        """
        Args:
            current_kspace: predicted kspace from the previous block
            ref_kspace: reference kspace for applying data consistency (is the under-sampled kspace in MRI reconstruction)
            mask: the under-sampling mask
            sens_maps: sensitivity maps for combining coil images
        """

        def sens_expand(x: Tensor) -> Tensor:
            """
            expands an image to its corresponding coil images based on the given sens_maps

            Args:
                x: image (B,1,H,W,2) with the last dimension being 2 (for real/imaginary parts)

            Returns:
                expansion of x to (B,num_coils,H,W,2) where num_coils is the number of coils.
            """
            return fftn_centered_t(complex_mul(x, sens_maps), spatial_dims=2)  # type: ignore

        def sens_reduce(x: Tensor) -> Tensor:
            """
            reduces coil images to a corresponding image based on the given sens_maps

            Args:
                x: kspace (B,num_coils,H,W,2) with the last dimension being 2 (for real/imaginary parts)

            Returns:
                reduction of x to (B,1,H,W,2)
            """
            x = ifftn_centered_t(x, spatial_dims=2)
            return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)  # type: ignore

        def soft_dc(x: Tensor) -> Tensor:
            """
            applies data consistency

            Args:
                x: kspace (B,num_coils,H,W,2) with the last dimension being 2 (for real/imaginary parts)
            """
            return torch.where(mask, x - ref_kspace, self.zeros) * self.dc_weight  # type: ignore

        return current_kspace - soft_dc(current_kspace) - sens_expand(self.model(sens_reduce(current_kspace)))
