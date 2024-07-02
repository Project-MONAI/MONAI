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

from monai.networks.layers.utils import get_rel_pos_embedding_layer
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
        hidden_input_size: int | None = None,
        dim_head: int | None = None,
        qkv_bias: bool = False,
        save_attn: bool = False,
        dim_head: int | None = None,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            hidden_input_size (int, optional): dimension of the input tensor. Defaults to hidden_size.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if dim_head:
            inner_dim = num_heads * dim_head
            self.dim_head = dim_head
        else:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden size should be divisible by num_heads.")
            inner_dim = hidden_size
            self.dim_head = hidden_size // num_heads

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads

        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.scale = self.dim_head**-0.5
        self.save_attn = save_attn
        self.attention_dtype = attention_dtype
        self.causal = causal
        self.sequence_length = sequence_length

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(sequence_length, sequence_length)).view(1, 1, sequence_length, sequence_length),
            )
            self.causal_mask: torch.Tensor

        self.att_mat = torch.Tensor()
        self.rel_positional_embedding = (
            get_rel_pos_embedding_layer(rel_pos_embedding, input_size, self.dim_head, self.num_heads)
            if rel_pos_embedding is not None
            else None
        )
        self.input_size = input_size

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C

        Return:
            torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
        """
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)
        att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale

        # apply relative positional embedding if defined
        att_mat = self.rel_positional_embedding(x, att_mat, q) if self.rel_positional_embedding is not None else att_mat

        if self.causal:
            att_mat = att_mat.masked_fill(self.causal_mask[:, :, : x.shape[1], : x.shape[1]] == 0, float("-inf"))

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
