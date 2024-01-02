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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
from monai.utils import optional_import


rearrange, _ = optional_import("einops", name="rearrange")


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        window_size: int = 0,
        input_size: Tuple = (),
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            window_size (int): Window size for local attention as used in Segment Anything https://arxiv.org/abs/2304.02643.
                If 0, global attention used. Only 2D inputs are supported for local attention (window_size > 0).
                See https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py.
            input_size (Tuple): spatial input dimensions (h, w, and d). Has to be set if local window attention is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.window_size = window_size
        self.input_size = input_size

    def forward(self, x):
        """
        Args:
            x (Tensor): [b x (s_dim_1 * … * s_dim_n) x dim]
        """
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            h, w = self.input_size
            x = rearrange(x, "b (h w) d -> b h w d", h=h, w=w)
            x, pad_hw = window_partition(x, self.window_size)
            x = rearrange(x, "b h w d -> b (h w) d", h=self.window_size, w=self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = rearrange(x, "b (h w) d -> b h w d", h=self.window_size, w=self.window_size)
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))
            x = rearrange(x, "b h w d -> b (h w) d", h=h, w=w)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed. Support only 2D.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    batch, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w

    x = x.view(batch, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (hp, wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    hp, wp = pad_hw
    h, w = hw
    batch = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(batch, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x
