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
"""
This script contains utility functions for developing new blocks in PyTorch.
"""

from typing import Sequence

from torch import Tensor

from monai.apps.reconstruction.complex_utils import complex_conj, complex_mul
from monai.networks.blocks.fft_utils_t import fftn_centered_t, ifftn_centered_t


def sens_reduce(x: Tensor, sens_maps: Tensor, spatial_dims: int = 2) -> Tensor:
    """
    Reduces coil measurements to a corresponding image based on the given sens_maps. Let's say there
    are C coil measurements inside x, then this function multiplies the conjugate of each coil sensitivity map with the
    corresponding coil image. The result of this process will be C images. Summing those images together gives the
    resulting "reduced image."

    Args:
        x: 2D kspace (B,C,H,W,2) with the last dimension being 2 (for real/imaginary parts) and C denoting the
            coil dimension. 3D data will have the shape (B,C,H,W,D,2).
        sens_maps: sensitivity maps of the same shape as input x.
        spatial_dims: is 2 for 2D data and is 3 for 3D data

    Returns:
        reduction of x to (B,1,H,W,2) for 2D data or (B,1,H,W,D,2) for 3D data.
    """
    x = ifftn_centered_t(x, spatial_dims=spatial_dims)  # inverse fourier transform
    return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)  # type: ignore


def sens_expand(x: Tensor, sens_maps: Tensor, spatial_dims: int = 2) -> Tensor:
    """
    Expands an image to its corresponding coil images based on the given sens_maps. Let's say there
    are C coils. This function multiples image x with each coil sensitivity map in sens_maps and stacks
    the resulting C coil images along the channel dimension which is reserved for coils.

    Args:
        x: 2D image (B,1,H,W,2) with the last dimension being 2 (for real/imaginary parts). 3D data will have
            the shape (B,1,H,W,D,2).
        sens_maps: Sensitivity maps for combining coil images. The shape is (B,C,H,W,2) for 2D data
            or (B,C,H,W,D,2) for 3D data (C denotes the coil dimension).
        spatial_dims: is 2 for 2D data and is 3 for 3D data

    Returns:
        Expansion of x to (B,C,H,W,2) for 2D data and (B,C,H,W,D,2) for 3D data. The output is transferred
            to the frquency domain to yield coil measurements.
    """
    return fftn_centered_t(complex_mul(x, sens_maps), spatial_dims=spatial_dims)  # type: ignore
