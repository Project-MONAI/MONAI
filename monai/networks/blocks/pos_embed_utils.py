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

from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, to_3tuple

__all__ = ["build_sincos_position_embedding"]


def build_sincos_position_embedding(
    grid_size: Optional[int], embed_dim: int, spatial_dims: int = 3, temperature: float = 10000.0
) -> torch.nn.Parameter:
    """
    Builds a sin-cos position embedding based on the given grid size, embed dimension, spatial dimensions, and temperature.
    Reference: https://github.com/cvlab-stonybrook/SelfMedMAE/blob/68d191dfcc1c7d0145db93a6a570362de29e3b30/lib/models/mae3d.py

    Args:
        grid_size (int or Tuple[int]): The size of the grid in each spatial dimension.
        embed_dim (int): The dimension of the embedding.
        spatial_dims (int): The number of spatial dimensions (2 for 2D, 3 for 3D).
        temperature (float): The temperature for the sin-cos position embedding.

    Returns:
        pos_embed (nn.Parameter): The sin-cos position embedding as a learnable parameter.
    """

    if spatial_dims == 2:
        grid_size = to_2tuple(grid_size)
        h, w = grid_size
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)

        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    elif spatial_dims == 3:
        grid_size = to_3tuple(grid_size)
        h, w, d = grid_size
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_d = torch.arange(d, dtype=torch.float32)

        grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d, indexing="ij")

        assert embed_dim % 6 == 0, "Embed dimension must be divisible by 6 for 3D sin-cos position embedding"

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
