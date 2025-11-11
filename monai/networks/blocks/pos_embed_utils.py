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

from __future__ import annotations

import collections.abc
from itertools import repeat
from typing import List, Union

import torch
import torch.nn as nn

__all__ = ["build_fourier_position_embedding", "build_sincos_position_embedding"]


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def build_fourier_position_embedding(
    grid_size: Union[int, List[int]], embed_dim: int, spatial_dims: int = 3, scales: Union[float, List[float]] = 1.0
) -> torch.nn.Parameter:
    """
    Builds a (Anistropic) Fourier feature position embedding based on the given grid size, embed dimension,
    spatial dimensions, and scales. The scales control the variance of the Fourier features, higher values make distant
    points more distinguishable.
    Position embedding is made anistropic by allowing setting different scales for each spatial dimension.
        Reference: https://arxiv.org/abs/2509.02488

    Args:
        grid_size (int | List[int]): The size of the grid in each spatial dimension.
        embed_dim (int): The dimension of the embedding.
        spatial_dims (int): The number of spatial dimensions (2 for 2D, 3 for 3D).
        scales (float | List[float]): The scale for every spatial dimension. If a single float is provided,
                              the same scale is used for all dimensions.

    Returns:
        pos_embed (nn.Parameter): The Fourier feature position embedding as a fixed parameter.
    """

    to_tuple = _ntuple(spatial_dims)
    grid_size_t = to_tuple(grid_size)
    if len(grid_size_t) != spatial_dims:
        raise ValueError(f"Length of grid_size ({len(grid_size_t)}) must be the same as spatial_dims.")

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for Fourier position embedding")

    # Ensure scales is a tensor of shape (spatial_dims,)
    if isinstance(scales, float):
        scales_tensor = torch.full((spatial_dims,), scales, dtype=torch.float)
    elif isinstance(scales, (list, tuple)):
        if len(scales) != spatial_dims:
            raise ValueError(f"Length of scales {len(scales)} does not match spatial_dims {spatial_dims}")
        scales_tensor = torch.tensor(scales, dtype=torch.float)
    else:
        raise TypeError(f"scales must be float or list of floats, got {type(scales)}")

    gaussians = torch.randn(embed_dim // 2, spatial_dims, dtype=torch.float32) * scales_tensor

    position_indices = [torch.linspace(0, 1, x, dtype=torch.float32) for x in grid_size_t]
    positions = torch.stack(torch.meshgrid(*position_indices, indexing="ij"), dim=-1)
    positions = positions.flatten(end_dim=-2)

    x_proj = (2.0 * torch.pi * positions) @ gaussians.T

    pos_emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    pos_emb = nn.Parameter(pos_emb[None, :, :], requires_grad=False)

    return pos_emb


def build_sincos_position_embedding(
    grid_size: Union[int, List[int]], embed_dim: int, spatial_dims: int = 3, temperature: float = 10000.0
) -> torch.nn.Parameter:
    """
    Builds a sin-cos position embedding based on the given grid size, embed dimension, spatial dimensions, and temperature.
    Reference: https://github.com/cvlab-stonybrook/SelfMedMAE/blob/68d191dfcc1c7d0145db93a6a570362de29e3b30/lib/models/mae3d.py

    Args:
        grid_size (List[int]): The size of the grid in each spatial dimension.
        embed_dim (int): The dimension of the embedding.
        spatial_dims (int): The number of spatial dimensions (2 for 2D, 3 for 3D).
        temperature (float): The temperature for the sin-cos position embedding.

    Returns:
        pos_embed (nn.Parameter): The sin-cos position embedding as a fixed parameter.
    """

    if spatial_dims == 2:
        to_2tuple = _ntuple(2)
        grid_size_t = to_2tuple(grid_size)
        h, w = grid_size_t
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)

        grid_h, grid_w = torch.meshgrid(grid_h, grid_w)

        if embed_dim % 4 != 0:
            raise AssertionError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    elif spatial_dims == 3:
        to_3tuple = _ntuple(3)
        grid_size_t = to_3tuple(grid_size)
        h, w, d = grid_size_t
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_d = torch.arange(d, dtype=torch.float32)

        grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)

        if embed_dim % 6 != 0:
            raise AssertionError("Embed dimension must be divisible by 6 for 3D sin-cos position embedding")

        pos_dim = embed_dim // 6
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
        pos_emb = torch.cat(
            [
                torch.sin(out_w),
                torch.cos(out_w),
                torch.sin(out_h),
                torch.cos(out_h),
                torch.sin(out_d),
                torch.cos(out_d),
            ],
            dim=1,
        )[None, :, :]
    else:
        raise NotImplementedError("Spatial Dimension Size {spatial_dims} Not Implemented!")

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False

    return pos_embed
