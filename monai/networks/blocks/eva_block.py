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

# Portions of this code are derived from the original repositories at:
# https://github.com/huggingface/pytorch-image-models
# https://github.com/MIC-DKFZ/dynamic-network-architectures
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.rope import apply_rotary_embedding
from monai.networks.layers import DropPath

__all__ = ["EVAAttention", "SwiGLUMLP", "EVABlock"]


class EVAAttention(nn.Module):
    """
    Multi-head self-attention as used in EVA-02: separate query/key/value projections (no key bias),
    optional rotary position embedding and an optional LayerNorm on the attention output before the projection.

    Args:
        hidden_size: token dimension.
        num_heads: number of attention heads.
        qkv_bias: whether the query and value projections have a bias.
        num_prefix_tokens: number of leading tokens (e.g. register tokens) that are not rotated.
        dropout_rate: dropout rate after the output projection.
        attention_dropout_rate: dropout rate on the attention weights.
        scale_norm: whether to apply a LayerNorm to the attention output.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        num_prefix_tokens: int = 0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        scale_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.num_prefix_tokens = num_prefix_tokens
        self.attention_dropout_rate = attention_dropout_rate
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm = nn.LayerNorm(hidden_size) if scale_norm else nn.Identity()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout_rate)

    def _rotate(self, t: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        """Apply ``rope`` to the non-prefix tokens of ``t`` of shape ``(B, num_heads, N, head_dim)``."""
        npt = self.num_prefix_tokens
        return torch.cat([t[:, :, :npt], apply_rotary_embedding(t[:, :, npt:], rope)], dim=2)

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: tokens of shape ``(B, N, hidden_size)``.
            rope: optional ``[sin, cos]`` rotary embedding for the ``N - num_prefix_tokens`` non-prefix tokens,
                of shape ``(N - num_prefix_tokens, 2 * head_dim)`` or ``(B, 1, N - num_prefix_tokens, 2 * head_dim)``.
        """
        b, n, c = x.shape
        q, k, v = (
            proj(x).reshape(b, n, self.num_heads, -1).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        if rope is not None:
            q, k = (self._rotate(t, rope).type_as(v) for t in (q, k))
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attention_dropout_rate if self.training else 0.0)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(self.norm(x)))


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network with separate gate and value projections and an optional
    LayerNorm on the hidden features (as in EVA-02).

    Args:
        hidden_size: input and output dimension.
        mlp_dim: hidden dimension.
        dropout_rate: dropout rate after the gating and after the output projection.
        scale_norm: whether to apply a LayerNorm to the hidden features.
    """

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, scale_norm: bool = True) -> None:
        super().__init__()
        self.fc1_g = nn.Linear(hidden_size, mlp_dim)
        self.fc1_x = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(mlp_dim) if scale_norm else nn.Identity()
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1_g(x)) * self.fc1_x(x))
        return self.drop2(self.fc2(self.norm(x)))


class EVABlock(nn.Module):
    """
    Pre-norm EVA-02 transformer block: rotary self-attention and a SwiGLU MLP, each with optional
    LayerScale and stochastic depth. Parameter names follow timm's ``EvaBlock`` with
    ``qkv_fused=False, swiglu_mlp=True``, so its state dicts load directly.

    Args:
        hidden_size: token dimension.
        num_heads: number of attention heads.
        mlp_ratio: ratio of the MLP hidden dimension to ``hidden_size``.
        qkv_bias: whether the query and value projections have a bias.
        scale_mlp: whether to apply a LayerNorm inside the MLP.
        scale_attn_inner: whether to apply a LayerNorm to the attention output.
        num_prefix_tokens: number of leading tokens that are not rotated.
        dropout_rate: dropout rate of the attention and MLP output projections.
        attention_dropout_rate: dropout rate on the attention weights.
        drop_path_rate: stochastic depth rate.
        init_values: initial LayerScale value, ``None`` disables LayerScale.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4 * 2 / 3,
        qkv_bias: bool = True,
        scale_mlp: bool = True,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = EVAAttention(
            hidden_size,
            num_heads,
            qkv_bias=qkv_bias,
            num_prefix_tokens=num_prefix_tokens,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            scale_norm=scale_attn_inner,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(hidden_size)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, int(hidden_size * mlp_ratio), dropout_rate=dropout_rate, scale_norm=scale_mlp)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(hidden_size)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None = None) -> torch.Tensor:
        attn = self.attn(self.norm1(x), rope=rope)
        if self.gamma_1 is not None:
            attn = self.gamma_1 * attn
        x = x + self.drop_path1(attn)
        mlp = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            mlp = self.gamma_2 * mlp
        return x + self.drop_path2(mlp)
