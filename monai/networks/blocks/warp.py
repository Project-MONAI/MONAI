import warnings
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from monai.config.deviceconfig import USE_COMPILED
from monai.networks.layers.spatial_transforms import grid_pull
from monai.utils import GridSamplePadMode

__all__ = ["Warp", "DVF2DDF"]


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
        # resolves _interp_mode for different methods
        if USE_COMPILED:
            self._interp_mode = mode
        else:
            warnings.warn("monai.networks.blocks.Warp: Using PyTorch native grid_sample.")
            self._interp_mode = "bilinear"
            if mode == 0:
                self._interp_mode = "nearest"
            elif mode == 1:
                self._interp_mode = "bilinear"
            elif mode == 3:
                self._interp_mode = "bicubic"  # torch.functional.grid_sample only supports 4D
            else:
                warnings.warn(f"{mode}-order interpolation is not supported, using linear interpolation.")

        # resolves _padding_mode for different methods
        padding_mode = GridSamplePadMode(padding_mode).value
        if USE_COMPILED:
            if padding_mode == "zeros":
                self._padding_mode = 7
            elif padding_mode == "border":
                self._padding_mode = 0
            else:
                self._padding_mode = 1  # reflection
        else:
            self._padding_mode = padding_mode

    @staticmethod
    def get_reference_grid(ddf: torch.Tensor) -> torch.Tensor:
        mesh_points = [torch.arange(0, dim) for dim in ddf.shape[2:]]
        grid = torch.stack(torch.meshgrid(*mesh_points), dim=0)  # (spatial_dims, ...)
        grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
        grid = grid.to(ddf)
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

        if not USE_COMPILED:  # pytorch native grid_sample
            for i, dim in enumerate(grid.shape[1:-1]):
                grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1
            index_ordering: List[int] = list(range(spatial_dims - 1, -1, -1))
            grid = grid[..., index_ordering]  # z, y, x -> x, y, z
            return F.grid_sample(
                image, grid, mode=self._interp_mode, padding_mode=self._padding_mode, align_corners=True
            )

        # using csrc resampling
        return grid_pull(
            image,
            grid,
            bound=self._padding_mode,
            extrapolate=True,
            interpolation=self._interp_mode,
        )


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
