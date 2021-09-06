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

from typing import Optional, Tuple, Union

import torch

from monai.transforms.transform import Transform

__all__ = ["SplitOnGrid"]


class SplitOnGrid(Transform):
    """
    Split the image into patches based on the provided grid shape.
    This transform works only with torch.Tensor inputs.

    Args:
        grid_shape: a tuple or an integer define the shape of the grid upon which to extract patches.
            If it's an integer, the value will be repeated for each dimension. Default is 2x2
        patch_size: a tuple or an integer that defines the output patch sizes.
            If it's an integer, the value will be repeated for each dimension.
            The default is (0, 0), where the patch size will be infered from the grid shape.

    Note: the shape of the input image is infered based on the first image used.
    """

    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]] = (2, 2),
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        # Grid size
        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size)
        else:
            self.grid_size = grid_size
        # Patch size
        self.patch_size = None
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.grid_size == (1, 1) and self.patch_size is None:
            return torch.stack([image])
        patch_size, steps = self.get_params(image.shape[1:])
        patches = (
            image.unfold(1, patch_size[0], steps[0])
            .unfold(2, patch_size[1], steps[1])
            .flatten(1, 2)
            .transpose(0, 1)
            .contiguous()
        )
        return patches

    def get_params(self, image_size):
        if self.patch_size is None:
            patch_size = tuple(image_size[i] // self.grid_size[i] for i in range(2))
        else:
            patch_size = self.patch_size

        steps = tuple(
            (image_size[i] - patch_size[i]) // (self.grid_size[i] - 1) if self.grid_size[i] > 1 else image_size[i]
            for i in range(2)
        )

        return patch_size, steps
