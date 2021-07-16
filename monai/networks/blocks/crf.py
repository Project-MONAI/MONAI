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

from typing import Optional

import torch
from torch.nn.functional import softmax

from monai.networks.layers.filtering import PHLFilter

__all__ = ["CRF"]


class CRF(torch.nn.Module):
    """
    Conditional Random Field: Combines message passing with a class
    compatibility convolution into an iterative process designed
    to successively minimise the energy of the class labeling.

    In this implementation, the message passing step is a weighted
    combination of a gaussian filter and a bilateral filter.
    The bilateral term is included to respect existing structure
    within the reference tensor.

    See:
        https://arxiv.org/abs/1502.03240
    """

    def __init__(
        self,
        iterations: int = 5,
        bilateral_weight: float = 1.0,
        gaussian_weight: float = 1.0,
        bilateral_spatial_sigma: float = 5.0,
        bilateral_color_sigma: float = 0.5,
        gaussian_spatial_sigma: float = 5.0,
        update_factor: float = 3.0,
        compatibility_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            iterations: the number of iterations.
            bilateral_weight: the weighting of the bilateral term in the message passing step.
            gaussian_weight: the weighting of the gaussian term in the message passing step.
            bilateral_spatial_sigma: standard deviation in spatial coordinates for the bilateral term.
            bilateral_color_sigma: standard deviation in color space for the bilateral term.
            gaussian_spatial_sigma: standard deviation in spatial coordinates for the gaussian term.
            update_factor: determines the magnitude of each update.
            compatibility_matrix: a matrix describing class compatibility,
                should be NxN where N is the number of classes.
        """
        super(CRF, self).__init__()
        self.iterations = iterations
        self.bilateral_weight = bilateral_weight
        self.gaussian_weight = gaussian_weight
        self.bilateral_spatial_sigma = bilateral_spatial_sigma
        self.bilateral_color_sigma = bilateral_color_sigma
        self.gaussian_spatial_sigma = gaussian_spatial_sigma
        self.update_factor = update_factor
        self.compatibility_matrix = compatibility_matrix

    def forward(self, input_tensor: torch.Tensor, reference_tensor: torch.Tensor):
        """
        Args:
            input_tensor: tensor containing initial class logits.
            reference_tensor: the reference tensor used to guide the message passing.

        Returns:
            output (torch.Tensor): output tensor.
        """

        # constructing spatial feature tensor
        spatial_features = _create_coordinate_tensor(reference_tensor)

        # constructing final feature tensors for bilateral and gaussian kernel
        bilateral_features = torch.cat(
            [spatial_features / self.bilateral_spatial_sigma, reference_tensor / self.bilateral_color_sigma], dim=1
        )
        gaussian_features = spatial_features / self.gaussian_spatial_sigma

        # setting up output tensor
        output_tensor = softmax(input_tensor, dim=1)

        # mean field loop
        for _ in range(self.iterations):

            # message passing step for both kernels
            bilateral_output = PHLFilter.apply(output_tensor, bilateral_features)
            gaussian_output = PHLFilter.apply(output_tensor, gaussian_features)

            # combining filter outputs
            combined_output = self.bilateral_weight * bilateral_output + self.gaussian_weight * gaussian_output

            # optionally running a compatibility transform
            if self.compatibility_matrix is not None:
                flat = combined_output.flatten(start_dim=2).permute(0, 2, 1)
                flat = torch.matmul(flat, self.compatibility_matrix)
                combined_output = flat.permute(0, 2, 1).reshape(combined_output.shape)

            # update and normalize
            output_tensor = softmax(input_tensor + self.update_factor * combined_output, dim=1)

        return output_tensor


# helper methods
def _create_coordinate_tensor(tensor):
    axes = [torch.arange(tensor.size(i)) for i in range(2, tensor.dim())]
    grids = torch.meshgrid(axes)
    coords = torch.stack(grids).to(device=tensor.device, dtype=tensor.dtype)
    return torch.stack(tensor.size(0) * [coords], dim=0)
