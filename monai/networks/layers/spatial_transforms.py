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

import torch
import torch.nn as nn

from monai.networks.utils import to_norm_affine

__all__ = ["AffineTransform"]


class AffineTransform(nn.Module):
    def __init__(
        self,
        theta,
        spatial_size=None,
        normalized=False,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
        reverse_indexing=True,
    ):
        """
        Apply affine transformations with a batch of affine matrices.

        `theta` must be an affine transformation matrix with shape
        3x3 or Nx3x3 or Nx2x3 or 2x3 for spatial 2D transforms,
        4x4 or Nx4x4 or Nx3x4 or 3x4 for spatial 3D transforms,
        where `N` is the batch size.

        When `normalized=False` and `reverse_indexing=True`,
        it does the commonly used resampling in the 'pull' direction
        following the `scipy.ndimage.affine_transform` convention.
        See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html

        When `normalized=True` and `reverse_indexing=False`,
        it applies `theta` to the normalized coordinates [-1, 1] directly.
        This is often used with `align_corners=False` to achieve resolution-agnostic resampling,
        thus useful as a part of trainable modules such as the spatial transformer networks.
        See also: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

        Args:
            theta (array_like): Nx3x3, Nx2x3, 3x3, 2x3 for spatial 2D inputs,
                Nx4x4, Nx3x4, 3x4, 4x4 for spatial 3D inputs.
            spatial_size (list or tuple of int): output spatial shape, the full output shape will be
                `[N, C, *spatial_size]` where N and C are inferred from the `src` input of `self.forward`.
            normalized (bool): indicating whether the provided affine matrix `theta` is defined
                for the normalized coordinates. If `normalized=False`, `theta` will be converted
                to operate on normalized coordinates as pytorch affine_grid works with the normalized
                coordinates.
            mode (`nearest|bilinear`): interpolation mode.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample.
            padding_mode (`zeros|border|reflection`): padding mode for outside grid values.
            align_corners (bool): see also https://pytorch.org/docs/stable/nn.functional.html#grid-sample.
            reverse_indexing (bool): whether to reverse the spatial indexing of image and coordinates.
                set to `False` if `theta` follows pytorch's default "D, H, W" convention.
                set to `True` if `theta` follows `scipy.ndimage` default "i, j, k" convention.
        """
        super().__init__()

        if spatial_size is not None:
            if not isinstance(spatial_size, (list, tuple)):
                raise ValueError("spatial size must be None or a list or tuple")
        self.theta = theta.float() if torch.is_tensor(theta) else torch.tensor(theta)
        if self.theta.ndim not in (2, 3):
            raise ValueError("affine must be Nxdxd or dxd.")
        if self.theta.ndim == 2:
            self.theta = self.theta[None]  # adds a batch dim.
        theta_shape = tuple(self.theta.shape[1:])
        if theta_shape in ((2, 3), (3, 4)):  # needs padding to dxd
            b = self.theta.shape[0]
            pad_affine = torch.tensor([0, 0, 1] if theta_shape[0] == 2 else [0, 0, 0, 1]).repeat(b, 1, 1)
            self.theta = torch.cat([self.theta, pad_affine.to(self.theta)], dim=1)
        if tuple(self.theta.shape[1:]) not in ((3, 3), (4, 4)):
            raise ValueError("affine must be Nx3x3 or Nx4x4, got: {}".format(self.theta.shape))

        self.spatial_size = spatial_size
        self.normalized = normalized
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.reverse_indexing = reverse_indexing

    def forward(self, src):
        """

        Args:
            src (array_like): image in spatial 2D (N, C, H, W) or spatial 3D (N, C, D, H, W),
                where N is the batch dim, C is the number of channels.
        """
        if not torch.is_tensor(src):
            # https://github.com/pytorch/pytorch/issues/33812
            # src = torch.tensor(src)
            src = torch.as_tensor(src)
        src = src.float()  # always use float for compatibility
        sr = src.ndim - 2  # input spatial rank
        if sr not in (2, 3):
            raise ValueError("src must be spatially 2D or 3D.")
        src_size = list(src.shape)
        if self.spatial_size is not None:
            dst_size = list(src_size[:2]) + list(self.spatial_size)
        else:
            dst_size = src_size  # default to the src shape

        self.theta = self.theta.to(src)
        theta = self.theta.clone()
        if self.reverse_indexing:
            rev_idx = torch.as_tensor(range(sr - 1, -1, -1), device=src.device)
            theta[:, :sr] = theta[:, rev_idx]
            theta[:, :, :sr] = theta[:, :, rev_idx]
        if not self.normalized:
            theta = to_norm_affine(
                affine=theta, src_size=src_size[2:], dst_size=dst_size[2:], align_corners=self.align_corners
            )
        if (theta.shape[0] == 1) and src_size[0] > 1:
            theta = theta.repeat(src_size[0], 1, 1)  # broadcast the batch dim.
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
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return dst
