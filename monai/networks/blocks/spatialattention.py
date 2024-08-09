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

from monai.networks.blocks import SABlock
from monai.utils import optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SpatialAttentionBlock(nn.Module):
    """Perform spatial self-attention on the input tensor.

    The input tensor is reshaped to B x (x_dim * y_dim [ * z_dim]) x C, where C is the number of channels, and then
    self-attention is performed on the reshaped tensor. The output tensor is reshaped back to the original shape.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        num_channels: number of input channels. Must be divisible by num_head_channels.
        num_head_channels: number of channels per head.
        norm_num_groups: Number of groups for the group norm layer.
        norm_eps: Epsilon for the normalization.
        attention_dtype: cast attention operations to this dtype.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).

    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        attention_dtype: Optional[torch.dtype] = None,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)
        # check num_head_channels is divisible by num_channels
        if num_head_channels is not None and num_channels % num_head_channels != 0:
            raise ValueError("num_channels must be divisible by num_head_channels")
        num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.attn = SABlock(
            hidden_size=num_channels,
            num_heads=num_heads,
            qkv_bias=True,
            attention_dtype=attention_dtype,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

    def forward(self, x: torch.Tensor):
        residual = x

        if self.spatial_dims == 1:
            h = x.shape[2]
            rearrange_input = Rearrange("b c h -> b h c")
            rearrange_output = Rearrange("b h c -> b c h", h=h)
        if self.spatial_dims == 2:
            h, w = x.shape[2], x.shape[3]
            rearrange_input = Rearrange("b c h w -> b (h w) c")
            rearrange_output = Rearrange("b (h w) c -> b c h w", h=h, w=w)
        else:
            h, w, d = x.shape[2], x.shape[3], x.shape[4]
            rearrange_input = Rearrange("b c h w d -> b (h w d) c")
            rearrange_output = Rearrange("b (h w d) c -> b c h w d", h=h, w=w, d=d)

        x = self.norm(x)
        x = rearrange_input(x)  # B x (x_dim * y_dim [ * z_dim]) x C

        x = self.attn(x)
        x = rearrange_output(x)  # B x  x C x x_dim * y_dim * [z_dim]
        x = x + residual
        return x
