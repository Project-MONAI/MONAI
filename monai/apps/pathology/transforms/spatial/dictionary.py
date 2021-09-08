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

from typing import Dict, Hashable, Mapping, Optional, Tuple, Union

import torch

from monai.config import KeysCollection
from monai.transforms.transform import MapTransform

from .array import SplitOnGrid

__all__ = ["SplitOnGridd", "SplitOnGridD", "SplitOnGridDict"]


class SplitOnGridd(MapTransform):
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
        keys: KeysCollection,
        grid_size: Union[int, Tuple[int, int]] = (2, 2),
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.splitter = SplitOnGrid(grid_size=grid_size, patch_size=patch_size)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.splitter(d[key])
        return d


SplitOnGridDict = SplitOnGridD = SplitOnGridd
