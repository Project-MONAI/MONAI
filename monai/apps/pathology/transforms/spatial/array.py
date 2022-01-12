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

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided

from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Randomizable, Transform
from monai.utils import convert_data_type, convert_to_dst_type
from monai.utils.enums import TransformBackends

__all__ = ["SplitOnGrid", "TileOnGrid"]


class SplitOnGrid(Transform):
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

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

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

    def __call__(self, image: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.grid_size == (1, 1) and self.patch_size is None:
            if isinstance(image, torch.Tensor):
                return torch.stack([image])
            elif isinstance(image, np.ndarray):
                return np.stack([image])  # type: ignore
            else:
                raise ValueError(f"Input type [{type(image)}] is not supported.")

        patch_size, steps = self.get_params(image.shape[1:])
        patches: NdarrayOrTensor
        if isinstance(image, torch.Tensor):
            patches = (
                image.unfold(1, patch_size[0], steps[0])
                .unfold(2, patch_size[1], steps[1])
                .flatten(1, 2)
                .transpose(0, 1)
                .contiguous()
            )
        elif isinstance(image, np.ndarray):
            x_step, y_step = steps
            c_stride, x_stride, y_stride = image.strides
            n_channels = image.shape[0]
            patches = as_strided(
                image,
                shape=(*self.grid_size, n_channels, patch_size[0], patch_size[1]),
                strides=(x_stride * x_step, y_stride * y_step, c_stride, x_stride, y_stride),
                writeable=False,
            )
            # flatten the first two dimensions
            patches = patches.reshape(np.prod(patches.shape[:2]), *patches.shape[2:])
            # make it a contiguous array
            patches = np.ascontiguousarray(patches)
        else:
            raise ValueError(f"Input type [{type(image)}] is not supported.")

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
        filter_mode: mode must be in ["min", "max", "random"]. If total number of tiles is more than tile_size,
            then sort by intensity sum, and take the smallest (for min), largest (for max) or random (for random) subset
            Defaults to ``min`` (which assumes background is high value)

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        tile_count: Optional[int] = None,
        tile_size: int = 256,
        step: Optional[int] = None,
        random_offset: bool = False,
        pad_full: bool = False,
        background_val: int = 255,
        filter_mode: str = "min",
    ):
        self.tile_count = tile_count
        self.tile_size = tile_size
        self.random_offset = random_offset
        self.pad_full = pad_full
        self.background_val = background_val
        self.filter_mode = filter_mode

        if step is None:
            # non-overlapping grid
            self.step = self.tile_size
        else:
            self.step = step

        self.offset = (0, 0)
        self.random_idxs = np.array((0,))

        if self.filter_mode not in ["min", "max", "random"]:
            raise ValueError("Unsupported filter_mode, must be [min, max or random]: " + str(self.filter_mode))

    def randomize(self, img_size: Sequence[int]) -> None:

        c, h, w = img_size

        self.offset = (0, 0)
        if self.random_offset:
            pad_h = h % self.tile_size
            pad_w = w % self.tile_size
            self.offset = (self.R.randint(pad_h) if pad_h > 0 else 0, self.R.randint(pad_w) if pad_w > 0 else 0)
            h = h - self.offset[0]
            w = w - self.offset[1]

        if self.pad_full:
            pad_h = (self.tile_size - h % self.tile_size) % self.tile_size
            pad_w = (self.tile_size - w % self.tile_size) % self.tile_size
            h = h + pad_h
            w = w + pad_w

        h_n = (h - self.tile_size + self.step) // self.step
        w_n = (w - self.tile_size + self.step) // self.step
        tile_total = h_n * w_n

        if self.tile_count is not None and tile_total > self.tile_count:
            self.random_idxs = self.R.choice(range(tile_total), self.tile_count, replace=False)
        else:
            self.random_idxs = np.array((0,))

    def __call__(self, image: NdarrayOrTensor) -> NdarrayOrTensor:
        img_np: np.ndarray
        img_np, *_ = convert_data_type(image, np.ndarray)  # type: ignore

        # add random offset
        self.randomize(img_size=img_np.shape)

        if self.random_offset and (self.offset[0] > 0 or self.offset[1] > 0):
            img_np = img_np[:, self.offset[0] :, self.offset[1] :]

        # pad to full size, divisible by tile_size
        if self.pad_full:
            c, h, w = img_np.shape
            pad_h = (self.tile_size - h % self.tile_size) % self.tile_size
            pad_w = (self.tile_size - w % self.tile_size) % self.tile_size
            img_np = np.pad(
                img_np,
                [[0, 0], [pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2]],
                constant_values=self.background_val,
            )

        # extact tiles
        x_step, y_step = self.step, self.step
        h_tile, w_tile = self.tile_size, self.tile_size
        c_image, h_image, w_image = img_np.shape
        c_stride, x_stride, y_stride = img_np.strides
        llw = as_strided(
            img_np,
            shape=((h_image - h_tile) // x_step + 1, (w_image - w_tile) // y_step + 1, c_image, h_tile, w_tile),
            strides=(x_stride * x_step, y_stride * y_step, c_stride, x_stride, y_stride),
            writeable=False,
        )
        img_np = llw.reshape(-1, c_image, h_tile, w_tile)

        # if keeping all patches
        if self.tile_count is None:
            # retain only patches with significant foreground content to speed up inference
            # FYI, this returns a variable number of tiles, so the batch_size must be 1 (per gpu), e.g during inference
            thresh = 0.999 * 3 * self.background_val * self.tile_size * self.tile_size
            if self.filter_mode == "min":
                # default, keep non-background tiles (small values)
                idxs = np.argwhere(img_np.sum(axis=(1, 2, 3)) < thresh)
                img_np = img_np[idxs.reshape(-1)]
            elif self.filter_mode == "max":
                idxs = np.argwhere(img_np.sum(axis=(1, 2, 3)) >= thresh)
                img_np = img_np[idxs.reshape(-1)]

        else:
            if len(img_np) > self.tile_count:

                if self.filter_mode == "min":
                    # default, keep non-background tiles (smallest values)
                    idxs = np.argsort(img_np.sum(axis=(1, 2, 3)))[: self.tile_count]
                    img_np = img_np[idxs]
                elif self.filter_mode == "max":
                    idxs = np.argsort(img_np.sum(axis=(1, 2, 3)))[-self.tile_count :]
                    img_np = img_np[idxs]
                else:
                    # random subset (more appropriate for WSIs without distinct background)
                    if self.random_idxs is not None:
                        img_np = img_np[self.random_idxs]

            elif len(img_np) < self.tile_count:
                img_np = np.pad(
                    img_np,
                    [[0, self.tile_count - len(img_np)], [0, 0], [0, 0], [0, 0]],
                    constant_values=self.background_val,
                )

        image, *_ = convert_to_dst_type(src=img_np, dst=image, dtype=image.dtype)

        return image
