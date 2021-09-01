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

from typing import TYPE_CHECKING, Optional, Tuple, Union

from torch.nn import Module

from monai.transforms.transform import Transform

if TYPE_CHECKING:
    import torch

__all__ = ["SplitOnGrid"]


class SplitOnGrid(Transform):
    """
    Split the image into patches based on the provided grid shape.
    This transform works only with torch.Tensor inputs.

    Args:
        grid_shape: a tuple or an integer define the shape of the grid upon which to extract patches.
            If it's an integer, th evalue will be repeated for each dimension. Default is 2x2
        patch_size: a tuple or an integer that defines the output patch sizes.
            If it's an integer, the value will be repeated for each dimension.
            If None (default), the patch size will be infered from the grid shape.
    """

    def __init__(
        self,
        grid_shape: Union[int, Tuple[int, int]] = (2, 2),
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        if isinstance(grid_shape, int):
            self.grid_shape = (grid_shape, grid_shape)
        else:
            self.grid_shape = grid_shape
        self.patch_size = None
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

    def __call__(self, region: torch.Tensor) -> torch.Tensor:
        _, h, w = region.shape
        if self.patch_size is None:
            if self.grid_shape == (1, 1):
                return region
            else:
                self.patch_size = (h // self.grid_shape[0], w // self.grid_shape[1])

        h_stride = (h - self.patch_size[0]) // (self.grid_shape[0] - 1)
        w_stride = (w - self.patch_size[1]) // (self.grid_shape[1] - 1)

        patches = (
            region.unfold(1, self.patch_size[0], h_stride)
            .unfold(2, self.patch_size[1], w_stride)
            .flatten(1, 2)
            .transpose(0, 1)
            .contiguous()
        )

        return patches
