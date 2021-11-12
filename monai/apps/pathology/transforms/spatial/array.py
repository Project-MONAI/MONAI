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

from typing import Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided

from monai.transforms.transform import Randomizable, Transform

__all__ = ["SplitOnGrid", "TileOnGrid"]


class SplitOnGrid(Transform):
    """
    Split the image into patches based on the provided grid shape.
    This transform works only with torch.Tensor inputs.

    Args:
        grid_shape: a tuple or an integer define the shape of the grid upon which to extract patches.
            If it's an integer, the value will be repeated for each dimension. Default is 2x2
        patch_size: a tuple or an integer that defines the output patch sizes.
            If it's an integer, the value will be repeated for each dimension.
            The default is (0, 0), where the patch size will be inferred from the grid shape.

    Note: the shape of the input image is inferred based on the first image used.
    """

    def __init__(
        self, grid_size: Union[int, Tuple[int, int]] = (2, 2), patch_size: Optional[Union[int, Tuple[int, int]]] = None
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


class TileOnGrid(Randomizable, Transform):
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
        filter_mode: mode must be in ["min", "max", "random", None]. If total number of tiles is more than tile_size,
            then sort by intensity sum, and take the smallest (for min), largest (for max) or random (for random or None) subset
            Defaults to ``min`` (which assumes background is high value)

    """

    def __init__(
        self,
        tile_count: Optional[int] = None,
        tile_size: int = 256,
        step: Optional[int] = None,
        random_offset: bool = False,
        pad_full: bool = False,
        background_val: int = 255,
        filter_mode: Optional[str] = "min",
    ):
        self.tile_count = tile_count
        self.tile_size = tile_size
        self.step = step
        self.random_offset = random_offset
        self.pad_full = pad_full
        self.background_val = background_val
        self.filter_mode = filter_mode

        if self.step is None:
            self.step = self.tile_size  # non-overlapping grid

        self.offset = (0, 0)
        self.random_idxs = np.array((0,))

    def randomize(self, img_size: Sequence[int]) -> None:

        c, h, w = img_size
        tile_step = cast(int, self.step)

        if self.random_offset:
            pad_h = h % self.tile_size
            pad_w = w % self.tile_size
            if pad_h > 0 and pad_w > 0:
                self.offset = (self.R.randint(pad_h), self.R.randint(pad_w))
                h = h - self.offset[0]
                w = w - self.offset[1]
            else:
                self.offset = (0, 0)

        if self.pad_full:
            pad_h = (self.tile_size - h % self.tile_size) % self.tile_size
            pad_w = (self.tile_size - w % self.tile_size) % self.tile_size
            h = h + pad_h
            w = w + pad_w

        h_n = (h - self.tile_size + tile_step) // tile_step
        w_n = (w - self.tile_size + tile_step) // tile_step
        tile_total = h_n * w_n

        if self.tile_count is not None and tile_total > self.tile_count:
            self.random_idxs = self.R.choice(range(tile_total), self.tile_count, replace=False)
        else:
            self.random_idxs = np.array((0,))

    def __call__(self, image: np.ndarray) -> np.ndarray:

        # add random offset
        self.randomize(img_size=image.shape)
        tile_step = cast(int, self.step)

        if self.random_offset and self.offset[0] > 0 and self.offset[1] > 0:
            image = image[:, self.offset[0] :, self.offset[1] :]

        # pad to full size, divisible by tile_size
        if self.pad_full:
            c, h, w = image.shape
            pad_h = (self.tile_size - h % self.tile_size) % self.tile_size
            pad_w = (self.tile_size - w % self.tile_size) % self.tile_size
            image = np.pad(
                image,
                [[0, 0], [pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2]],
                constant_values=self.background_val,
            )

        # extact tiles
        xstep, ystep = tile_step, tile_step
        xsize, ysize = self.tile_size, self.tile_size
        clen, xlen, ylen = image.shape
        cstride, xstride, ystride = image.strides
        llw = as_strided(
            image,
            shape=((xlen - xsize) // xstep + 1, (ylen - ysize) // ystep + 1, clen, xsize, ysize),
            strides=(xstride * xstep, ystride * ystep, cstride, xstride, ystride),
            writeable=False,
        )
        image = llw.reshape(-1, clen, xsize, ysize)

        # if keeping all patches
        if self.tile_count is None:
            # retain only patches with significant foreground content to speed up inference
            # FYI, this returns a variable number of tiles, so the batch_size must be 1 (per gpu), e.g during inference
            thresh = 0.999 * 3 * self.background_val * self.tile_size * self.tile_size
            if self.filter_mode == "min":
                # default, keep non-background tiles (small values)
                idxs = np.argwhere(image.sum(axis=(1, 2, 3)) < thresh)
                image = image[idxs.reshape(-1)]
            elif self.filter_mode == "max":
                idxs = np.argwhere(image.sum(axis=(1, 2, 3)) >= thresh)
                image = image[idxs.reshape(-1)]

        else:
            if len(image) >= self.tile_count:

                if self.filter_mode == "min":
                    # default, keep non-background tiles (smallest values)
                    idxs = np.argsort(image.sum(axis=(1, 2, 3)))[: self.tile_count]
                    image = image[idxs]
                elif self.filter_mode == "max":
                    idxs = np.argsort(image.sum(axis=(1, 2, 3)))[-self.tile_count :]
                    image = image[idxs]
                elif len(image) > self.tile_count:
                    # random subset (more appropriate for WSIs without distinct background)
                    if self.random_idxs is not None:
                        image = image[self.random_idxs]

            else:
                image = np.pad(
                    image,
                    [[0, self.tile_count - len(image)], [0, 0], [0, 0], [0, 0]],
                    constant_values=self.background_val,
                )

        return image
