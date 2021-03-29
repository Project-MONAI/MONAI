from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from monai.utils import GridSamplePadMode


class Warp(nn.Module):
    """
    Warp an image with given dense displacement field (DDF).
    """

    def __init__(
        self,
        mode: int = 1,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.ZEROS,
    ):
        """
        Args:
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
        # (batch, ..., spatial_dims)
        for i, dim in enumerate(grid.shape[1:-1]):
            grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1
        return grid

    def forward(self, image: torch.Tensor, ddf: torch.Tensor) -> torch.Tensor:
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
            index_ordering: List[int] = list(range(spatial_dims - 1, -1, -1))
            grid = grid[..., index_ordering]  # z, y, x -> x, y, z
            _interp_mode = "bilinear" if self.mode == 1 else "nearest"
            warped_image = F.grid_sample(
                image, grid, mode=_interp_mode, padding_mode=self.padding_mode.value, align_corners=True
            )

        return warped_image


class DVF2DDF(nn.Module):
    """
    Layer calculates a dense velocity field (DVF) from a dense displacement field (DDF)
    with scaling and squaring.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)

    """

    def __init__(
        self,
        num_steps: int = 7,
        mode: int = 1,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.ZEROS,
    ):
        super(DVF2DDF, self).__init__()
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
        ddf: torch.Tensor = dvf / (2 ** self.num_steps)
        for _ in range(self.num_steps):
            ddf = ddf + self.warp_layer(image=ddf, ddf=ddf)
        return ddf
