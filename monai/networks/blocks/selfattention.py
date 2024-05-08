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
import warnings

import torch
import torch.nn as nn

from monai.networks.layers.utils import get_rel_pos_embedding_layer
from monai.networks.blocks.attention_utils import window_partition, window_unpartition
from monai.utils import optional_import

xops, has_xformers = optional_import("xformers.ops")
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        rel_pos_embedding: Optional[str] = None,
        input_size: Optional[Tuple] = None,
        causal: bool = False,
        sequence_length: int | None = None,
        use_flash_attention: bool = False,
        window_size: int = 0,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            rel_pos_embedding (str, optional): Add relative positional embeddings to the attention map.
                For now only "decomposed" is supported (see https://arxiv.org/abs/2112.01526). 2D and 3D are supported.
            input_size (tuple(spatial_dim), optional): Input resolution for calculating the relative
                positional parameter size. Has to be set if local window attention is used
            causal (bool): wether to use causal attention. If true `sequence_length` has to be set
            sequence_length (int, optional): if causal is True, it is necessary to specify the sequence length.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            window_size (int): Window size for local attention as used in Segment Anything https://arxiv.org/abs/2304.02643.
                If 0, global attention used.
                See https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        if use_flash_attention and rel_pos_embedding is not None:
            self.use_flash_attention = False
            warnings.warn(
                "flash attention set to `False`: flash attention can't be used with relative position embedding. Set `rel_pos_embedding` to `None` to use flash attention"
            )
        else:
            self.use_flash_attention = use_flash_attention

        if use_flash_attention and not has_xformers:
            raise ValueError("use_flash_attention is True but xformers is not installed.")
        if window_size > 0 and len(input_size) not in [2, 3]:
            raise ValueError(
                "If local window attention is used (window_size > 0), input_size should be specified: (h, w) or (h, w, d)"
            )

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.dropout_rate = dropout_rate
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.causal = causal
        self.sequence_length = sequence_length
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()
        self.rel_positional_embedding = (
            get_rel_pos_embedding_layer(
                rel_pos_embedding,
                [window_size] * len(input_size) if window_size > 0 else input_size,
                self.head_dim,
                self.num_heads,
            )
            if rel_pos_embedding is not None
            else None
        )
        self.window_size = window_size
        self.input_size = input_size

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(sequence_length, sequence_length)).view(1, 1, sequence_length, sequence_length),
            )
            self.causal_mask: torch.Tensor

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C

        Return:
            torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
        """

        if self.window_size > 0:
            x, pad = window_partition(x, self.window_size, self.input_size)

        _, t, _ = x.size()
        output = self.input_rearrange(self.qkv(x))  # 3 x B x (s_dim_1 * ... * s_dim_n) x h x C/h
        q, k, v = output[0], output[1], output[2]

        if self.use_flash_attention:
            x = xops.memory_efficient_attention(
                query=q.contiguous(),
                key=k.contiguous(),
                value=v.contiguous(),
                scale=self.scale,
                p=self.dropout_rate,
                attn_bias=xops.LowerTriangularMask() if self.causal else None,
            )
        else:
            att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale

            # apply relative positional embedding if defined
            att_mat = (
                self.rel_positional_embedding(x, att_mat, q) if self.rel_positional_embedding is not None else att_mat
            )
            # apply causal mask if set
            att_mat = (
                att_mat.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf")) if self.causal else att_mat
            )

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

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad, self.input_size)

        return x
