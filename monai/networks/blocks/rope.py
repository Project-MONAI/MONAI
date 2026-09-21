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

# Portions of this code are derived from the original repository at:
# https://github.com/huggingface/pytorch-image-models
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["SpatialRotaryEmbedding", "apply_rotary_embedding"]


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    # [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rotary_embedding(x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension of ``x`` by a rotary position embedding.

    Args:
        x: tensor of shape ``(..., seq_len, head_dim)``.
        embedding: concatenated ``[sin, cos]`` embedding of shape ``(seq_len, 2 * head_dim)``, or with
            additional leading dimensions broadcastable against ``x``, as returned by
            :py:class:`SpatialRotaryEmbedding`.
    """
    sin, cos = embedding.chunk(2, -1)
    return x * cos + _rotate_interleaved(x) * sin


class SpatialRotaryEmbedding(nn.Module):
    """
    Axial rotary position embedding (RoPE) over an N-dimensional token grid, as used by EVA-02 style
    vision transformers.

    Each spatial axis gets ``head_dim // (2 * len(feat_shape))`` inverse-frequency bands; the per-axis
    angles are concatenated and interleaved so that consecutive channel pairs of a head are rotated together.
    The embedding for the flattened grid (row-major, "ij" indexing) is cached as a non-persistent buffer.

    Args:
        head_dim: per-head channel dimension the embedding is applied to. Must be divisible by
            ``2 * len(feat_shape)``.
        feat_shape: spatial shape of the token grid, e.g. ``(8, 8, 8)``.
        temperature: base of the inverse frequencies.
    """

    embedding: torch.Tensor

    def __init__(self, head_dim: int, feat_shape: Sequence[int], temperature: float = 10000.0) -> None:
        super().__init__()
        spatial_dims = len(feat_shape)
        if spatial_dims < 1:
            raise ValueError("feat_shape must have at least one spatial dimension.")
        if head_dim % (2 * spatial_dims) != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 2 * len(feat_shape) ({2 * spatial_dims}).")
        self.head_dim = head_dim
        self.feat_shape = tuple(int(s) for s in feat_shape)
        self.temperature = temperature
        self.register_buffer("embedding", self._build(), persistent=False)

    def _build(self) -> torch.Tensor:
        num_bands = self.head_dim // (2 * len(self.feat_shape))
        bands = 1.0 / (self.temperature ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))
        coords = [torch.arange(s, dtype=torch.float32) for s in self.feat_shape]
        grid = torch.stack(torch.meshgrid(coords, indexing="ij"), dim=-1)  # (*feat_shape, spatial_dims)
        angles = (grid.unsqueeze(-1) * bands).reshape(-1, len(self.feat_shape) * num_bands)
        angles = angles.repeat_interleave(2, -1)  # (num_tokens, head_dim)
        return torch.cat([angles.sin(), angles.cos()], -1)

    def forward(self) -> torch.Tensor:
        """Return the cached ``[sin, cos]`` embedding of shape ``(num_tokens, 2 * head_dim)``."""
        return self.embedding
