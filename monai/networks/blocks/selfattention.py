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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.utils import optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    One can setup relative positional embedding as described in <https://arxiv.org/abs/2112.01526>
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        use_rel_pos: Optional[str] = None,
        input_size: Optional[Tuple] = None,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            rel_pos (str, optional): Add relative positional embeddings to the attention map.
                For now only "decomposed" is supported (see https://arxiv.org/abs/2112.01526). 2D and 3D are supported.
            input_size (tuple(spatial_dim), optional): Input resolution for calculating the relative
                positional parameter size.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        if self.use_rel_pos == "decomposed":
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_arr = nn.ParameterList(
                [nn.Parameter(torch.zeros(2 * dim_input_size - 1, self.head_dim)) for dim_input_size in input_size]
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C

        Return:
            torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
        """
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale

        if self.use_rel_pos == "decomposed":
            batch = x.shape[0]
            h, w = self.input_size[:2] if self.input_size is not None else (0, 0)
            d = self.input_size[2] if self.input_size is not None and len(self.input_size) > 2 else 1
            att_mat = add_decomposed_rel_pos(
                att_mat.view(batch * self.num_heads, h * w * d, h * w * d),
                q.view(batch * self.num_heads, h * w * d, -1),
                self.rel_pos_arr,
                (h, w) if d == 1 else (h, w, d),
                (h, w) if d == 1 else (h, w, d),
            )
            att_mat = att_mat.reshape(batch, self.num_heads, h * w * d, h * w * d)

        att_mat = att_mat.softmax(dim=-1)

        if self.save_attn:
            # no gradients and new tensor;
            # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    rel_pos_resized: torch.Tensor = torch.Tensor()
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear"
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor, q: torch.Tensor, rel_pos_lst: nn.ParameterList, q_size: Tuple, k_size: Tuple
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Only 2D and 3D are supported.
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, s_dim_1 * ... * s_dim_n, C).
        rel_pos_lst (ParameterList): relative position embeddings for each axis: rel_pos_lst[n] for nth axis.
        q_size (Tuple): spatial sequence size of query q with (q_dim_1, ..., q_dim_n).
        k_size (Tuple): spatial sequence size of key k with (k_dim_1, ...,  k_dim_n).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    rh = get_rel_pos(q_size[0], k_size[0], rel_pos_lst[0])
    rw = get_rel_pos(q_size[1], k_size[1], rel_pos_lst[1])

    batch, _, dim = q.shape

    if len(rel_pos_lst) == 2:
        q_h, q_w = q_size[:2]
        k_h, k_w = k_size[:2]
        r_q = q.reshape(batch, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

        attn = (attn.view(batch, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
            batch, q_h * q_w, k_h * k_w
        )
    elif len(rel_pos_lst) == 3:
        q_h, q_w, q_d = q_size[:3]
        k_h, k_w, k_d = k_size[:3]

        rd = get_rel_pos(q_d, k_d, rel_pos_lst[2])

        r_q = q.reshape(batch, q_h, q_w, q_d, dim)
        rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, rh)
        rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, rw)
        rel_d = torch.einsum("bhwdc,wkc->bhwdk", r_q, rd)

        attn = (
            attn.view(batch, q_h, q_w, q_d, k_h, k_w, k_d)
            + rel_h[:, :, :, :, None, None]
            + rel_w[:, :, :, None, :, None]
            + rel_d[:, :, :, None, None, :]
        ).view(batch, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn
