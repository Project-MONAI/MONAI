# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch
import torch.nn as nn

from monai.networks import to_norm_affine
from monai.utils import ensure_tuple, GridSampleMode, GridSamplePadMode

__all__ = ["AffineTransform"]


class AffineTransform(nn.Module):
    def __init__(
        self,
        spatial_size=None,
        normalized: bool = False,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.ZEROS,
        align_corners: bool = False,
        reverse_indexing: bool = True,
    ):
        """
        Apply affine transformations with a batch of affine matrices.

        When `normalized=False` and `reverse_indexing=True`,
        it does the commonly used resampling in the 'pull' direction
        following the ``scipy.ndimage.affine_transform`` convention.
        In this case `theta` is equivalent to (ndim+1, ndim+1) input ``matrix`` of ``scipy.ndimage.affine_transform``,
        operates on homogeneous coordinates.
        See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html

        When `normalized=True` and `reverse_indexing=False`,
        it applies `theta` to the normalized coordinates (coords. in the range of [-1, 1]) directly.
        This is often used with `align_corners=False` to achieve resolution-agnostic resampling,
        thus useful as a part of trainable modules such as the spatial transformer networks.
        See also: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

        Args:
            spatial_size (list or tuple of int): output spatial shape, the full output shape will be
                `[N, C, *spatial_size]` where N and C are inferred from the `src` input of `self.forward`.
            normalized: indicating whether the provided affine matrix `theta` is defined
                for the normalized coordinates. If `normalized=False`, `theta` will be converted
                to operate on normalized coordinates as pytorch affine_grid works with the normalized
                coordinates.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: see also https://pytorch.org/docs/stable/nn.functional.html#grid-sample.
            reverse_indexing: whether to reverse the spatial indexing of image and coordinates.
                set to `False` if `theta` follows pytorch's default "D, H, W" convention.
                set to `True` if `theta` follows `scipy.ndimage` default "i, j, k" convention.
        """
        super().__init__()
        self.spatial_size = ensure_tuple(spatial_size) if spatial_size is not None else None
        self.normalized = normalized
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.align_corners = align_corners
        self.reverse_indexing = reverse_indexing

    def forward(self, src, theta, spatial_size=None):
        """
        ``theta`` must be an affine transformation matrix with shape
        3x3 or Nx3x3 or Nx2x3 or 2x3 for spatial 2D transforms,
        4x4 or Nx4x4 or Nx3x4 or 3x4 for spatial 3D transforms,
        where `N` is the batch size. `theta` will be converted into float Tensor for the computation.

        Args:
            src (array_like): image in spatial 2D or 3D (N, C, spatial_dims),
                where N is the batch dim, C is the number of channels.
            theta (array_like): Nx3x3, Nx2x3, 3x3, 2x3 for spatial 2D inputs,
                Nx4x4, Nx3x4, 3x4, 4x4 for spatial 3D inputs. When the batch dimension is omitted,
                `theta` will be repeated N times, N is the batch dim of `src`.
            spatial_size (list or tuple of int): output spatial shape, the full output shape will be
                `[N, C, *spatial_size]` where N and C are inferred from the `src`.

        Raises:
            TypeError: both src and theta must be torch Tensor, got {type(src).__name__}, {type(theta).__name__}.
            ValueError: affine must be Nxdxd or dxd.
            ValueError: affine must be Nx3x3 or Nx4x4, got: {theta.shape}.
            ValueError: src must be spatially 2D or 3D.
            ValueError: batch dimension of affine and image does not match, got affine: {} and image: {}.

        """
        # validate `theta`
        if not torch.is_tensor(theta) or not torch.is_tensor(src):
            raise TypeError(
                f"both src and theta must be torch Tensor, got {type(src).__name__}, {type(theta).__name__}."
            )
        if theta.ndim not in (2, 3):
            raise ValueError("affine must be Nxdxd or dxd.")
        if theta.ndim == 2:
            theta = theta[None]  # adds a batch dim.
        theta = theta.clone()  # no in-place change of theta
        theta_shape = tuple(theta.shape[1:])
        if theta_shape in ((2, 3), (3, 4)):  # needs padding to dxd
            pad_affine = torch.tensor([0, 0, 1] if theta_shape[0] == 2 else [0, 0, 0, 1])
            pad_affine = pad_affine.repeat(theta.shape[0], 1, 1).to(theta)
            pad_affine.requires_grad = False
            theta = torch.cat([theta, pad_affine], dim=1)
        if tuple(theta.shape[1:]) not in ((3, 3), (4, 4)):
            raise ValueError(f"affine must be Nx3x3 or Nx4x4, got: {theta.shape}.")

        # validate `src`
        sr = src.ndim - 2  # input spatial rank
        if sr not in (2, 3):
            raise ValueError("src must be spatially 2D or 3D.")

        # set output shape
        src_size = tuple(src.shape)
        dst_size = src_size  # default to the src shape
        if self.spatial_size is not None:
            dst_size = src_size[:2] + self.spatial_size
        if spatial_size is not None:
            dst_size = src_size[:2] + ensure_tuple(spatial_size)

        # reverse and normalise theta if needed
        if not self.normalized:
            theta = to_norm_affine(
                affine=theta, src_size=src_size[2:], dst_size=dst_size[2:], align_corners=self.align_corners
            )
        if self.reverse_indexing:
            rev_idx = torch.as_tensor(range(sr - 1, -1, -1), device=src.device)
            theta[:, :sr] = theta[:, rev_idx]
            theta[:, :, :sr] = theta[:, :, rev_idx]
        if (theta.shape[0] == 1) and src_size[0] > 1:
            # adds a batch dim to `theta` in order to match `src`
            theta = theta.repeat(src_size[0], 1, 1)
        if theta.shape[0] != src_size[0]:
            raise ValueError(
                "batch dimension of affine and image does not match, got affine: {} and image: {}.".format(
                    theta.shape[0], src_size[0]
                )
            )

        grid = nn.functional.affine_grid(theta=theta[:, :sr], size=dst_size, align_corners=self.align_corners)
        dst = nn.functional.grid_sample(
            input=src.contiguous(),
            grid=grid,
            mode=self.mode.value,
            padding_mode=self.padding_mode.value,
            align_corners=self.align_corners,
        )
        return dst
