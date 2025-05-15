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

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import trunc_normal_
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option

SUPPORTED_POS_EMBEDDING_TYPES = {"none", "learnable", "sincos"}

__all__ = ["MaskedAutoEncoderViT"]


class MaskedAutoEncoderViT(nn.Module):
    """
    Masked Autoencoder (ViT), based on: "Kaiming et al.,
    Masked Autoencoders Are Scalable Vision Learners <https://arxiv.org/abs/2111.06377>"
    Only a subset of the patches passes through the encoder. The decoder tries to reconstruct
    the masked patches, resulting in improved training speed.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 12,
        masking_ratio: float = 0.75,
        decoder_hidden_size: int = 384,
        decoder_mlp_dim: int = 512,
        decoder_num_layers: int = 4,
        decoder_num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "sincos",
        decoder_pos_embed_type: str = "sincos",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input.
            img_size: dimension of input image.
            patch_size: dimension of patch size
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 512.
            num_layers:  number of transformer blocks. Defaults to 12.
            num_heads: number of attention heads. Defaults to 12.
            masking_ratio: ratio of patches to be masked. Defaults to 0.75.
            decoder_hidden_size: dimension of hidden layer for decoder. Defaults to 384.
            decoder_mlp_dim: dimension of feedforward layer for decoder. Defaults to 512.
            decoder_num_layers: number of transformer blocks for decoder. Defaults to 4.
            decoder_num_heads: number of attention heads for decoder. Defaults to 12.
            proj_type: position embedding layer type. Defaults to "conv".
            pos_embed_type: position embedding layer type. Defaults to "sincos".
            decoder_pos_embed_type: position embedding layer type for decoder. Defaults to "sincos".
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dimensions. Defaults to 3.
            qkv_bias: apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
        Examples::
            # for single channel input with image size of (96,96,96), and sin-cos positional encoding
            >>> net = MaskedAutoEncoderViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16),
            pos_embed_type='sincos')
            # for 3-channel with image size of (128,128,128) and a learnable positional encoding
            >>> net = MaskedAutoEncoderViT(in_channels=3, img_size=128, patch_size=16, pos_embed_type='learnable')
            # for 3-channel with image size of (224,224) and a masking ratio of 0.25
            >>> net = MaskedAutoEncoderViT(in_channels=3, img_size=(224,224), patch_size=(16,16), masking_ratio=0.25,
            spatial_dims=2)
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate should be between 0 and 1, got {dropout_rate}.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        if decoder_hidden_size % decoder_num_heads != 0:
            raise ValueError("decoder_hidden_size should be divisible by decoder_num_heads.")

        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.spatial_dims = spatial_dims
        for m, p in zip(self.img_size, self.patch_size):
            if m % p != 0:
                raise ValueError(f"patch_size={patch_size} should be divisible by img_size={img_size}.")

        self.decoder_hidden_size = decoder_hidden_size

        if masking_ratio <= 0 or masking_ratio >= 1:
            raise ValueError(f"masking_ratio should be in the range (0, 1), got {masking_ratio}.")

        self.masking_ratio = masking_ratio
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        blocks = [
            TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
            for _ in range(num_layers)
        ]
        self.blocks = nn.Sequential(*blocks, nn.LayerNorm(hidden_size))

        # decoder
        self.decoder_embed = nn.Linear(hidden_size, decoder_hidden_size)

        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

        self.decoder_pos_embed_type = look_up_option(decoder_pos_embed_type, SUPPORTED_POS_EMBEDDING_TYPES)
        self.decoder_pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches, decoder_hidden_size))

        decoder_blocks = [
            TransformerBlock(decoder_hidden_size, decoder_mlp_dim, decoder_num_heads, dropout_rate, qkv_bias, save_attn)
            for _ in range(decoder_num_layers)
        ]
        self.decoder_blocks = nn.Sequential(*decoder_blocks, nn.LayerNorm(decoder_hidden_size))
        self.decoder_pred = nn.Linear(decoder_hidden_size, int(np.prod(self.patch_size)) * in_channels)

        self._init_weights()

    def _init_weights(self):
        """
        similar to monai/networks/blocks/patchembedding.py for the decoder positional encoding and for mask and
        classification tokens
        """
        if self.decoder_pos_embed_type == "none":
            pass
        elif self.decoder_pos_embed_type == "learnable":
            trunc_normal_(self.decoder_pos_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.decoder_pos_embed_type == "sincos":
            grid_size = []
            for in_size, pa_size in zip(self.img_size, self.patch_size):
                grid_size.append(in_size // pa_size)

            self.decoder_pos_embedding = build_sincos_position_embedding(
                grid_size, self.decoder_hidden_size, self.spatial_dims
            )

        else:
            raise ValueError(f"decoder_pos_embed_type {self.decoder_pos_embed_type} not supported.")

        # initialize patch_embedding like nn.Linear (instead of nn.Conv2d)
        trunc_normal_(self.mask_tokens, mean=0.0, std=0.02, a=-2.0, b=2.0)
        trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def _masking(self, x, masking_ratio: float | None = None):
        batch_size, num_tokens, _ = x.shape
        percentage_to_keep = 1 - masking_ratio if masking_ratio is not None else 1 - self.masking_ratio
        selected_indices = torch.multinomial(
            torch.ones(batch_size, num_tokens), int(percentage_to_keep * num_tokens), replacement=False
        )
        x_masked = x[torch.arange(batch_size).unsqueeze(1), selected_indices]  # gather the selected tokens
        mask = torch.ones(batch_size, num_tokens, dtype=torch.int).to(x.device)
        mask[torch.arange(batch_size).unsqueeze(-1), selected_indices] = 0

        return x_masked, selected_indices, mask

    def forward(self, x, masking_ratio: float | None = None):
        x = self.patch_embedding(x)
        x, selected_indices, mask = self._masking(x, masking_ratio=masking_ratio)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)

        # decoder
        x = self.decoder_embed(x)

        x_ = self.mask_tokens.repeat(x.shape[0], mask.shape[1], 1)
        x_[torch.arange(x.shape[0]).unsqueeze(-1), selected_indices] = x[:, 1:, :]  # no cls token
        x_ = x_ + self.decoder_pos_embedding
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = self.decoder_blocks(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]
        return x, mask
