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

import warnings
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from monai.config.deviceconfig import USE_COMPILED
from monai.networks.layers.spatial_transforms import grid_pull
from monai.networks.utils import meshgrid_ij
from monai.utils import GridSampleMode, GridSamplePadMode, optional_import

_C, _ = optional_import("monai._C")

__all__ = ["Warp", "DVF2DDF"]


class Warp(nn.Module):
    """
    Warp an image with given dense displacement field (DDF).
    """

    def __init__(self, mode=GridSampleMode.BILINEAR.value, padding_mode=GridSamplePadMode.BORDER.value):
        """
        For pytorch native APIs, the possible values are:

            - mode: ``"nearest"``, ``"bilinear"``, ``"bicubic"``.
            - padding_mode: ``"zeros"``, ``"border"``, ``"reflection"``

        See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

        For MONAI C++/CUDA extensions, the possible values are:

            - mode: ``"nearest"``, ``"bilinear"``, ``"bicubic"``, 0, 1, ...
            - padding_mode: ``"zeros"``, ``"border"``, ``"reflection"``, 0, 1, ...

        See also: :py:class:`monai.networks.layers.grid_pull`
        """
        super().__init__()
        # resolves _interp_mode for different methods

        if USE_COMPILED:
            if mode in (inter.value for inter in GridSampleMode):
                mode = GridSampleMode(mode)
                if mode == GridSampleMode.BILINEAR:
                    mode = 1
                elif mode == GridSampleMode.NEAREST:
                    mode = 0
                elif mode == GridSampleMode.BICUBIC:
                    mode = 3
                else:
                    mode = 1  # default to linear
            self._interp_mode = mode
        else:
            warnings.warn("monai.networks.blocks.Warp: Using PyTorch native grid_sample.")
            self._interp_mode = GridSampleMode(mode).value

        # resolves _padding_mode for different methods
        if USE_COMPILED:
            if padding_mode in (pad.value for pad in GridSamplePadMode):
                padding_mode = GridSamplePadMode(padding_mode)
                if padding_mode == GridSamplePadMode.ZEROS:
                    padding_mode = 7
                elif padding_mode == GridSamplePadMode.BORDER:
                    padding_mode = 0
                elif padding_mode == GridSamplePadMode.REFLECTION:
                    padding_mode = 1
                else:
                    padding_mode = 0  # default to nearest
            self._padding_mode = padding_mode
        else:
            self._padding_mode = GridSamplePadMode(padding_mode).value

    @staticmethod
    def get_reference_grid(ddf: torch.Tensor) -> torch.Tensor:
        mesh_points = [torch.arange(0, dim) for dim in ddf.shape[2:]]
        grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
        grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
        grid = grid.to(ddf)
        return grid

    def forward(self, image: torch.Tensor, ddf: torch.Tensor):
        """
        Args:
            image: Tensor in shape (batch, num_channels, H, W[, D])
            ddf: Tensor in the same spatial size as image, in shape (batch, ``spatial_dims``, H, W[, D])

        Returns:
            warped_image in the same shape as image (batch, num_channels, H, W[, D])
        """
        spatial_dims = len(image.shape) - 2
        if spatial_dims not in (2, 3):
            raise NotImplementedError(f"got unsupported spatial_dims={spatial_dims}, currently support 2 or 3.")
        ddf_shape = (image.shape[0], spatial_dims) + tuple(image.shape[2:])
        if ddf.shape != ddf_shape:
            raise ValueError(
                f"Given input {spatial_dims}-d image shape {image.shape}, " f"the input DDF shape must be {ddf_shape}."
            )
        grid = self.get_reference_grid(ddf) + ddf
        grid = grid.permute([0] + list(range(2, 2 + spatial_dims)) + [1])  # (batch, ..., spatial_dims)

        if not USE_COMPILED:  # pytorch native grid_sample
            for i, dim in enumerate(grid.shape[1:-1]):
                grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1
            index_ordering: List[int] = list(range(spatial_dims - 1, -1, -1))
            grid = grid[..., index_ordering]  # z, y, x -> x, y, z
            return F.grid_sample(
                image, grid, mode=self._interp_mode, padding_mode=f"{self._padding_mode}", align_corners=True
            )

        # using csrc resampling
        return grid_pull(image, grid, bound=self._padding_mode, extrapolate=True, interpolation=self._interp_mode)


class DVF2DDF(nn.Module):
    """
    Layer calculates a dense displacement field (DDF) from a dense velocity field (DVF)
    with scaling and squaring.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)

    """

    def __init__(
        self, num_steps: int = 7, mode=GridSampleMode.BILINEAR.value, padding_mode=GridSamplePadMode.ZEROS.value
    ):
        super().__init__()
        if num_steps <= 0:
            raise ValueError(f"expecting positive num_steps, got {num_steps}")
        self.num_steps = num_steps
        self.warp_layer = Warp(mode=mode, padding_mode=padding_mode)

    def forward(self, dvf):
        """
        Args:
            dvf: dvf to be transformed, in shape (batch, ``spatial_dims``, H, W[,D])

        Returns:
            a dense displacement field
        """
        ddf: torch.Tensor = dvf / (2**self.num_steps)
        for _ in range(self.num_steps):
            ddf = ddf + self.warp_layer(image=ddf, ddf=ddf)
        return ddf
