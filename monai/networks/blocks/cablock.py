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

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.utils import optional_import

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = ["FeedForward", "CABlock"]


class FeedForward(nn.Module):
    """Gated-DConv Feed-Forward Network (GDFN) that controls feature flow using gating mechanism.
    Uses depth-wise convolutions for local context mixing and GELU-activated gating for refined feature selection.

    Args:
        spatial_dims: Number of spatial dimensions (2D or 3D)
        dim: Number of input channels
        ffn_expansion_factor: Factor to expand hidden features dimension
        bias: Whether to use bias in convolution layers
    """

    def __init__(self, spatial_dims: int, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=dim,
            out_channels=hidden_features * 2,
            kernel_size=1,
            bias=bias,
            conv_only=True,
        )

        self.dwconv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=hidden_features * 2,
            out_channels=hidden_features * 2,
            kernel_size=3,
            strides=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
            conv_only=True,
        )

        self.project_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=hidden_features,
            out_channels=dim,
            kernel_size=1,
            bias=bias,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return cast(torch.Tensor, self.project_out(F.gelu(x1) * x2))


class CABlock(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA): Differs from standard self-attention
    by operating on feature channels instead of spatial dimensions. Incorporates depth-wise
    convolutions for local mixing before attention, achieving linear complexity vs quadratic
    in vanilla attention. Based on SW Zamir, et al., 2022 <https://arxiv.org/abs/2111.09881>

    Args:
        spatial_dims: Number of spatial dimensions (2D or 3D)
        dim: Number of input channels
        num_heads: Number of attention heads
        bias: Whether to use bias in convolution layers
        flash_attention: Whether to use flash attention optimization. Defaults to False.

    Raises:
        ValueError: If flash attention is not available in current PyTorch version
        ValueError: If spatial_dims is greater than 3
    """

    def __init__(self, spatial_dims, dim: int, num_heads: int, bias: bool, flash_attention: bool = False):
        super().__init__()
        if flash_attention and not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("Flash attention not available")
        if spatial_dims > 3:
            raise ValueError(f"Only 2D and 3D inputs are supported. Got spatial_dims={spatial_dims}")
        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.flash_attention = flash_attention

        self.qkv = Convolution(
            spatial_dims=spatial_dims, in_channels=dim, out_channels=dim * 3, kernel_size=1, bias=bias, conv_only=True
        )

        self.qkv_dwconv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=dim * 3,
            out_channels=dim * 3,
            kernel_size=3,
            strides=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
            conv_only=True,
        )

        self.project_out = Convolution(
            spatial_dims=spatial_dims, in_channels=dim, out_channels=dim, kernel_size=1, bias=bias, conv_only=True
        )

        self._attention_fn = self._get_attention_fn()

    def _get_attention_fn(self):
        if self.flash_attention:
            return self._flash_attention
        return self._normal_attention

    def _flash_attention(self, q, k, v):
        """Flash attention implementation using scaled dot-product attention."""
        scale = float(self.temperature.mean())
        out = F.scaled_dot_product_attention(q, k, v, scale=scale, dropout_p=0.0, is_causal=False)
        return out

    def _normal_attention(self, q, k, v):
        """Attention matrix multiplication with depth-wise convolutions."""
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        return attn @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MDTA attention.
        1. Apply depth-wise convolutions to Q, K, V
        2. Reshape Q, K, V for multi-head attention
        3. Compute attention matrix using flash or normal attention
        4. Reshape and project out attention output"""
        spatial_dims = x.shape[2:]

        # Project and mix
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Select attention
        if self.spatial_dims == 2:
            qkv_to_multihead = "b (head c) h w -> b head c (h w)"
            multihead_to_qkv = "b head c (h w) -> b (head c) h w"
        else:  # dims == 3
            qkv_to_multihead = "b (head c) d h w -> b head c (d h w)"
            multihead_to_qkv = "b head c (d h w) -> b (head c) d h w"

        # Reconstruct and project feature map
        q = rearrange(q, qkv_to_multihead, head=self.num_heads)
        k = rearrange(k, qkv_to_multihead, head=self.num_heads)
        v = rearrange(v, qkv_to_multihead, head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        out = self._attention_fn(q, k, v)
        out = rearrange(
            out,
            multihead_to_qkv,
            head=self.num_heads,
            **dict(zip(["h", "w"] if self.spatial_dims == 2 else ["d", "h", "w"], spatial_dims)),
        )

        return cast(torch.Tensor, self.project_out(out))
