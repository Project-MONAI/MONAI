from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from monai.config import USE_COMPILED
from monai.networks.layers import grid_pull
from monai.utils import GridSampleMode, GridSamplePadMode


class Warp(nn.Module):
    """
    Warp an image with given DDF.
    """

    def __init__(
        self,
        spatial_dims: int,
        mode: Optional[Union[GridSampleMode, str]] = GridSampleMode.BILINEAR,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.ZEROS,
    ):
        """
        Args:
            spatial_dims: {2, 3}. number of spatial dimensions
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        super(Warp, self).__init__()
        if spatial_dims not in [2, 3]:
            raise ValueError(f"got unsupported spatial_dims = {spatial_dims}, only support 2-d and 3-d input")
        self.spatial_dims = spatial_dims
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)

    @staticmethod
    def get_reference_grid(ddf: torch.Tensor) -> torch.Tensor:
        mesh_points = [torch.arange(0, dim) for dim in ddf.shape[2:]]
        grid = torch.stack(torch.meshgrid(*mesh_points[::-1]), dim=0)  # (spatial_dims, ...)
        grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
        grid = grid.to(ddf)
        return grid

    @staticmethod
    def normalize_grid(grid: torch.Tensor) -> torch.Tensor:
        # (batch, ..., self.spatial_dims)
        for i, dim in enumerate(grid.shape[1:-1]):
            grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1
        return grid

    def forward(self, image: torch.Tensor, ddf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor in shape (batch, num_channels, H, W[, D])
            ddf: Tensor in the same spatial size as image, in shape (batch, spatial_dims, H, W[, D])

        Returns:
            warped_image in the same shape as image (batch, num_channels, H, W[, D])
        """
        if len(image.shape) != 2 + self.spatial_dims:
            raise ValueError(f"expecting {self.spatial_dims + 2}-d input, " f"got input in shape {image.shape}")
        if len(ddf.shape) != 2 + self.spatial_dims or ddf.shape[1] != self.spatial_dims:
            raise ValueError(
                f"expecting {self.spatial_dims + 2}-d ddf with {self.spatial_dims} channels, "
                f"got ddf in shape {ddf.shape}"
            )
        if image.shape[0] != ddf.shape[0] or image.shape[2:] != ddf.shape[2:]:
            raise ValueError(
                "expecting image and ddf of same batch size and spatial size, "
                f"got image of shape {image.shape}, ddf of shape {ddf.shape}"
            )

        grid = self.get_reference_grid(ddf) + ddf
        grid = grid.permute([0] + list(range(2, 2 + self.spatial_dims)) + [1])  # (batch, ..., self.spatial_dims)

        if USE_COMPILED:
            _padding_mode = self.padding_mode.value
            if _padding_mode == "zeros":
                bound = 7
            elif _padding_mode == "border":
                bound = 0
            else:
                bound = 1
            _interp_mode = self.mode.value
            warped_image: torch.Tensor = grid_pull(
                image,
                grid,
                bound=bound,
                extrapolate=True,
                interpolation=1 if _interp_mode == "bilinear" else _interp_mode,
            )
        else:
            grid = self.normalize_grid(grid)
            index_ordering: List[int] = list(range(self.spatial_dims - 1, -1, -1))
            grid = grid[..., index_ordering]  # z, y, x -> x, y, z
            warped_image = F.grid_sample(
                image, grid, mode=self.mode.value, padding_mode=self.padding_mode.value, align_corners=True
            )
        return warped_image
