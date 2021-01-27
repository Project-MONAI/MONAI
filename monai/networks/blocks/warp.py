from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from monai.utils import GridSamplePadMode


class Warp(nn.Module):
    """
    Warp an image with given DDF.
    """

    def __init__(
        self,
        spatial_dims: int,
        mode: int = 1,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.ZEROS,
    ):
        """
        Args:
            spatial_dims: {2, 3}. number of spatial dimensions
            mode: interpolation mode to calculate output values, defaults to 1.
                Possible values are::

                    - 0 or 'nearest'    or InterpolationType.nearest
                    - 1 or 'linear'     or InterpolationType.linear
                    - 2 or 'quadratic'  or InterpolationType.quadratic
                    - 3 or 'cubic'      or InterpolationType.cubic
                    - 4 or 'fourth'     or InterpolationType.fourth
                    - etc.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        super(Warp, self).__init__()
        if spatial_dims not in [2, 3]:
            raise ValueError(f"got unsupported spatial_dims={spatial_dims}, only support 2-d and 3-d input")
        self.spatial_dims = spatial_dims
        if mode < 0:
            raise ValueError(f"do not support negative mode, got mode={mode}")
        self.mode = mode
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)

    @staticmethod
    def get_reference_grid(ddf: torch.Tensor) -> torch.Tensor:
        mesh_points = [torch.arange(0, dim) for dim in ddf.shape[2:]]
        grid = torch.stack(torch.meshgrid(*mesh_points), dim=0)  # (spatial_dims, ...)
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

        if self.mode > 1:
            raise ValueError(f"{self.mode}-order interpolation not yet implemented.")
            # if not USE_COMPILED:
            #     raise ValueError(f"cannot perform {self.mode}-order interpolation without C compile.")
            # _padding_mode = self.padding_mode.value
            # if _padding_mode == "zeros":
            #     bound = 7
            # elif _padding_mode == "border":
            #     bound = 0
            # else:
            #     bound = 1
            # warped_image: torch.Tensor = grid_pull(
            #     image,
            #     grid,
            #     bound=bound,
            #     extrapolate=True,
            #     interpolation=self.mode,
            # )
        else:
            grid = self.normalize_grid(grid)
            index_ordering: List[int] = list(range(self.spatial_dims - 1, -1, -1))
            grid = grid[..., index_ordering]  # z, y, x -> x, y, z
            _interp_mode = "bilinear" if self.mode == 1 else "nearest"
            warped_image = F.grid_sample(
                image, grid, mode=_interp_mode, padding_mode=self.padding_mode.value, align_corners=True
            )

        return warped_image
