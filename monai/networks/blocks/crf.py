# Copyright 2020 MONAI Consortium
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
from torch.nn.functional import conv1d, conv2d, conv3d, pad, softmax

from monai.networks.layers.filtering import PHLFilter

__all__ = ["CRF"]


class CRF(torch.nn.Module):
    """
    Conditional Random Field: Combines message passing with a class
    compatability convolution into an iterative process designed
    to successively minimise the energy of the class labeling.

    In this implementation, the message passing step is a weighted
    combination of a gaussian filter and a bilateral filter.
    The bilateral term is included to respect existing structure
    within the reference tensor.

    See:
        https://arxiv.org/abs/1502.03240

    Args:
        input_tensor: tensor containing initial class logits.

        referenece_tensor: the reference tensor used to guide the message passing.

        bilateral_weight: the weighting of the bilateral term in the message passing step

        gaussian_weight: the weighting of the gaussian term in the message passing step

        bilateral_spatial_sigma: standard deviation in spatial coordinates for the bilateral term

        bilateral_color_sigma: standard deviation in color space for the bilateral term

        gaussian_spatial_sigma: standard deviation in spatial coordinates for the gaussian term

        compatability_kernel_range: the range of the kernel used in the compatability convolution

        iterations: the number of iterations.

    Returns:
        output (torch.Tensor): output tensor.
    """

    def __init__(
        self,
        bilateral_weight: [float] = 0.8,
        gaussian_weight: [float] = 0.2,
        bilateral_spatial_sigma: [float] = 64,
        bilateral_color_sigma: [float] = 0.2,
        gaussian_spatial_sigma: [float] = 64,
        compatability_kernel_range: [int] = 1,
        iterations: [int] = 5,
    ):
        super(CRF, self).__init__()
        self.bilateral_weight = bilateral_weight
        self.gaussian_weight = gaussian_weight
        self.bilateral_spatial_sigma = bilateral_spatial_sigma
        self.bilateral_color_sigma = bilateral_color_sigma
        self.gaussian_spatial_sigma = gaussian_spatial_sigma
        self.compatability_kernel_range = compatability_kernel_range
        self.iterations = iterations

    def forward(self, input_tensor: torch.Tensor, reference_tensor: torch.Tensor):

        # useful values
        spatial_dim = input_tensor.dim() - 2
        class_count = input_tensor.size(1)
        padding = self.compatability_kernel_range

        # constructing spatial feature tensor
        spatial_features = _create_coordinate_tensor(reference_tensor)

        # constructing final feature tensors for bilateral and gaussian kernel
        bilateral_features = torch.cat(
            [spatial_features / self.bilateral_spatial_sigma, reference_tensor / self.bilateral_color_sigma], dim=1
        )
        gaussian_features = spatial_features / self.gaussian_spatial_sigma

        # compatability matrix (potts model (1 - diag) for now)
        compatability_matrix = _potts_model_weights(class_count).to(device=input_tensor.device)

        # expanding matrix to kernel
        compatability_kernel = _expand_matrix_to_kernel(
            compatability_matrix, spatial_dim, self.compatability_kernel_range
        )

        # choosing convolution function
        conv = [conv1d, conv2d, conv3d][spatial_dim - 1]

        # seting up output tensor
        output_tensor = softmax(input_tensor, dim=1)

        # mean field loop
        for _ in range(self.iterations):

            # message passing step for both kernels
            bliateral_output = PHLFilter.apply(output_tensor, bilateral_features)
            gaussian_output = PHLFilter.apply(output_tensor, gaussian_features)

            # combining filter outputs
            combined_output = self.bilateral_weight * bliateral_output + self.gaussian_weight * gaussian_output
            combined_output /= self.bilateral_weight + self.gaussian_weight

            # compatibility convolution
            combined_output = pad(combined_output, 2 * spatial_dim * [padding], mode="replicate")
            compatibility_update = conv(combined_output, compatability_kernel)

            # update and normalize
            output_tensor = softmax(input_tensor - compatibility_update, dim=1)

        return output_tensor


# helper methods
def _create_coordinate_tensor(tensor):
    axes = [torch.arange(tensor.size(i)) for i in range(2, tensor.dim())]
    grids = torch.meshgrid(axes)
    return torch.stack(grids).unsqueeze(0).to(device=tensor.device, dtype=tensor.dtype)


def _potts_model_weights(class_count):
    return (1 - torch.diag(torch.ones(class_count))).unsqueeze(-1)


def _expand_matrix_to_kernel(matrix, spatial_dim, kernel_range):
    reshape_arg = (matrix.size(0), matrix.size(1)) + spatial_dim * (1,)
    expand_arg = (-1, -1) + spatial_dim * (1 + 2 * kernel_range,)
    return matrix.reshape(reshape_arg).expand(expand_arg)
