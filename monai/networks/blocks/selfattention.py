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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.utils import get_rel_pos_embedding_layer
from monai.utils import optional_import, pytorch_after

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
        dim_head: int | None = None,
        hidden_input_size: int | None = None,
        causal: bool = False,
        sequence_length: int | None = None,
        rel_pos_embedding: str | None = None,
        input_size: Tuple | None = None,
        attention_dtype: torch.dtype | None = None,
        include_fc: bool = True,
        use_combined_linear: bool = True,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.
            hidden_input_size (int, optional): dimension of the input tensor. Defaults to hidden_size.
            causal: whether to use causal attention (see https://arxiv.org/abs/1706.03762).
            sequence_length: if causal is True, it is necessary to specify the sequence length.
            rel_pos_embedding (str, optional): Add relative positional embeddings to the attention map.
                For now only "decomposed" is supported (see https://arxiv.org/abs/2112.01526). 2D and 3D are supported.
            input_size (tuple(spatial_dim), optional): Input resolution for calculating the relative
                positional parameter size.
            attention_dtype: cast attention operations to this dtype.
            include_fc: whether to include the final linear layer. Default to True.
            use_combined_linear: whether to use a single linear layer for qkv projection, default to True.
            use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
                (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        if dim_head:
            self.inner_dim = num_heads * dim_head
            self.dim_head = dim_head
        else:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden size should be divisible by num_heads.")
            self.inner_dim = hidden_size
            self.dim_head = hidden_size // num_heads

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        if use_flash_attention and not pytorch_after(minor=13, major=1, patch=0):
            raise ValueError(
                "use_flash_attention is only supported for PyTorch versions >= 2.0."
                "Upgrade your PyTorch or set the flag to False."
            )
        if use_flash_attention and save_attn:
            raise ValueError(
                "save_attn has been set to True, but use_flash_attention is also set"
                "to True. save_attn can only be used if use_flash_attention is False."
            )

        if use_flash_attention and rel_pos_embedding is not None:
            raise ValueError("rel_pos_embedding must be None if you are using flash_attention.")

        self.num_heads = num_heads
        self.hidden_input_size = hidden_input_size if hidden_input_size else hidden_size
        self.out_proj = nn.Linear(self.inner_dim, self.hidden_input_size)

        self.qkv: Union[nn.Linear, nn.Identity]
        self.to_q: Union[nn.Linear, nn.Identity]
        self.to_k: Union[nn.Linear, nn.Identity]
        self.to_v: Union[nn.Linear, nn.Identity]

        if use_combined_linear:
            self.qkv = nn.Linear(self.hidden_input_size, self.inner_dim * 3, bias=qkv_bias)
            self.to_q = self.to_k = self.to_v = nn.Identity()  # add to enable torchscript
            self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        else:
            self.to_q = nn.Linear(self.hidden_input_size, self.inner_dim, bias=qkv_bias)
            self.to_k = nn.Linear(self.hidden_input_size, self.inner_dim, bias=qkv_bias)
            self.to_v = nn.Linear(self.hidden_input_size, self.inner_dim, bias=qkv_bias)
            self.qkv = nn.Identity()  # add to enable torchscript
            self.input_rearrange = Rearrange("b h (l d) -> b l h d", l=num_heads)
        self.out_rearrange = Rearrange("b l h d -> b h (l d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.scale = self.dim_head**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()
        self.attention_dtype = attention_dtype
        self.causal = causal
        self.sequence_length = sequence_length
        self.include_fc = include_fc
        self.use_combined_linear = use_combined_linear
        self.use_flash_attention = use_flash_attention

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(sequence_length, sequence_length)).view(1, 1, sequence_length, sequence_length),
            )
            self.causal_mask: torch.Tensor
        else:
            self.causal_mask = torch.Tensor()

        self.rel_positional_embedding = (
            get_rel_pos_embedding_layer(rel_pos_embedding, input_size, self.dim_head, self.num_heads)
            if rel_pos_embedding is not None
            else None
        )
        self.input_size = input_size

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C
            attn_mask (torch.Tensor, optional): mask to apply to the attention matrix.
            B x (s_dim_1 * ... * s_dim_n). Defaults to None.

        Return:
            torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
        """
        if self.use_combined_linear:
            output = self.input_rearrange(self.qkv(x))
            q, k, v = output[0], output[1], output[2]
        else:
            q = self.input_rearrange(self.to_q(x))
            k = self.input_rearrange(self.to_k(x))
            v = self.input_rearrange(self.to_v(x))

        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)

        if self.use_flash_attention:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                scale=self.scale,
                dropout_p=self.dropout_rate,
                is_causal=self.causal,
            )
        else:
            att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale

            # apply relative positional embedding if defined
            if self.rel_positional_embedding is not None:
                att_mat = self.rel_positional_embedding(x, att_mat, q)

            if self.causal:
                if attn_mask is not None:
                    raise ValueError("Causal attention does not support attention masks.")
                att_mat = att_mat.masked_fill(self.causal_mask[:, :, : x.shape[-2], : x.shape[-2]] == 0, float("-inf"))

            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
                att_mat = att_mat.masked_fill(attn_mask == 0, float("-inf"))

            att_mat = att_mat.softmax(dim=-1)
            if self.save_attn:
                # no gradients and new tensor;
                # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                self.att_mat = att_mat.detach()

            att_mat = self.drop_weights(att_mat)
            x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)

        x = self.out_rearrange(x)
        if self.include_fc:
            x = self.out_proj(x)
        x = self.drop_output(x)
        return x
