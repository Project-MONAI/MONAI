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

from monai.apps.reconstruction.networks.blocks.utils import sens_expand, sens_reduce


class VarNetBlock(nn.Module):
    """
    A variational block based on Sriram et. al., "End-to-end variational networks for accelerated MRI reconstruction".
    It applies data consistency and refinement to the intermediate kspace and combines those results.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        refinement_model: the model used for refinement (typically a U-Net but can be any deep learning model
            that performs well when the input and output are in image domain (e.g., a convolutional network).
        spatial_dims: is 2 for 2D data and is 3 for 3D data
    """

    def __init__(self, refinement_model: nn.Module, spatial_dims: int = 2):
        super().__init__()
        self.model = refinement_model
        self.spatial_dims = spatial_dims
        self.dc_weight = nn.Parameter(torch.ones(1))  # learned scalar as the multiplier of the DC block

        buffer_shape = [1 for _ in range(spatial_dims + 3)]  # 3 denotes the batch, channel, and real/complex dimensions
        self.register_buffer("zeros", torch.zeros(buffer_shape))

    def soft_dc(self, x: Tensor, ref_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Applies data consistency to input x. Suppose x is some intermediate estimate of the kspace and ref_kspace
        is the reference under-sampled measurement. This function returns mask * (x - ref_kspace). View this as the
        residual between the original under-sampled kspace and the estimate given by the network.

        Args:
            x: 2D kspace (B,C,H,W,2) with the last dimension being 2 (for real/imaginary parts) and C denoting the
                coil dimension. 3D data will have the shape (B,C,H,W,D,2).
            ref_kspace: original under-sampled kspace with the same shape as x.
            mask: the under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.

        Returns:
            Output of DC block with the same shape as x
        """
        return torch.where(mask, x - ref_kspace, self.zeros) * self.dc_weight  # type: ignore

    def forward(self, current_kspace: Tensor, ref_kspace: Tensor, mask: Tensor, sens_maps: Tensor) -> Tensor:
        """
        Args:
            current_kspace: Predicted kspace from the previous block. It's a 2D kspace (B,C,H,W,2)
                with the last dimension being 2 (for real/imaginary parts) and C denoting the
                coil dimension. 3D data will have the shape (B,C,H,W,D,2).
            ref_kspace: reference kspace for applying data consistency (is the under-sampled kspace in MRI reconstruction).
                Its shape is the same as current_kspace.
            mask: the under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.
            sens_maps: coil sensitivity maps with the same shape as current_kspace

        Returns:
            Output of VarNetBlock with the same shape as current_kspace
        """
        dc_out = soft_dc(current_kspace, ref_kspace, mask)  # output of DC block
        refinement_out = sens_expand(
            self.model(sens_reduce(current_kspace, sens_maps, spatial_dims=self.spatial_dims)),
            sens_maps,
            spatial_dims=self.spatial_dims,
        )  # output of refinement model
        output = current_kspace - dc_out - refinement_out
        return output
