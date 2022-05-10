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

import copy
from typing import Any, Dict, Hashable, List, Mapping, Optional, Tuple, Union

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, Randomizable

from .array import SplitOnGrid, TileOnGrid

__all__ = ["SplitOnGridd", "SplitOnGridD", "SplitOnGridDict", "TileOnGridd", "TileOnGridD", "TileOnGridDict"]


class SplitOnGridd(MapTransform):
    """
    Split the image into patches based on the provided grid shape.
    This transform works only with torch.Tensor inputs.

    Args:
        grid_size: a tuple or an integer define the shape of the grid upon which to extract patches.
            If it's an integer, the value will be repeated for each dimension. Default is 2x2
        patch_size: a tuple or an integer that defines the output patch sizes.
            If it's an integer, the value will be repeated for each dimension.
            The default is (0, 0), where the patch size will be inferred from the grid shape.

    Note: the shape of the input image is inferred based on the first image used.
    """

    backend = SplitOnGrid.backend

    def __init__(
        self,
        keys: KeysCollection,
        grid_size: Union[int, Tuple[int, int]] = (2, 2),
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.splitter = SplitOnGrid(grid_size=grid_size, patch_size=patch_size)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.splitter(d[key])
        return d


class TileOnGridd(Randomizable, MapTransform):
    """
    Tile the 2D image into patches on a grid and maintain a subset of it.
    This transform works only with np.ndarray inputs for 2D images.

    Args:
        tile_count: number of tiles to extract, if None extracts all non-background tiles
            Defaults to ``None``.
        tile_size: size of the square tile
            Defaults to ``256``.
        step: step size
            Defaults to ``None`` (same as tile_size)
        random_offset: Randomize position of the grid, instead of starting from the top-left corner
            Defaults to ``False``.
        pad_full: pad image to the size evenly divisible by tile_size
            Defaults to ``False``.
        background_val: the background constant (e.g. 255 for white background)
            Defaults to ``255``.
        filter_mode: mode must be in ["min", "max", "random"]. If total number of tiles is more than tile_size,
            then sort by intensity sum, and take the smallest (for min), largest (for max) or random (for random) subset
            Defaults to ``min`` (which assumes background is high value)

    """

    backend = SplitOnGrid.backend

    def __init__(
        self,
        keys: KeysCollection,
        tile_count: Optional[int] = None,
        tile_size: int = 256,
        step: Optional[int] = None,
        random_offset: bool = False,
        pad_full: bool = False,
        background_val: int = 255,
        filter_mode: str = "min",
        allow_missing_keys: bool = False,
        return_list_of_dicts: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.return_list_of_dicts = return_list_of_dicts
        self.seed = None

        self.splitter = TileOnGrid(
            tile_count=tile_count,
            tile_size=tile_size,
            step=step,
            random_offset=random_offset,
            pad_full=pad_full,
            background_val=background_val,
            filter_mode=filter_mode,
        )

    def randomize(self, data: Any = None) -> None:
        self.seed = self.R.randint(10000)  # type: ignore

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Union[Dict[Hashable, NdarrayOrTensor], List[Dict[Hashable, NdarrayOrTensor]]]:

        self.randomize()

        d = dict(data)
        for key in self.key_iterator(d):
            self.splitter.set_random_state(seed=self.seed)  # same random seed for all keys
            d[key] = self.splitter(d[key])

        if self.return_list_of_dicts:
            d_list = []
            for i in range(len(d[self.keys[0]])):
                d_list.append({k: d[k][i] if k in self.keys else copy.deepcopy(d[k]) for k in d.keys()})
            d = d_list  # type: ignore

        return d


SplitOnGridDict = SplitOnGridD = SplitOnGridd
TileOnGridDict = TileOnGridD = TileOnGridd
