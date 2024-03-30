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
from monai.utils import ensure_tuple_rep, is_sqrt
from monai.utils.module import look_up_option

SUPPORTED_POS_EMBEDDING_TYPES = {"none", "learnable", "sincos"}

__all__ = ["MaskedAutoEncoderViT"]


class MaskedAutoEncoderViT(nn.Module):
    """
    Masked Autoencoder (ViT), based on: "Kaiming et al.,
    Masked Autoencoders Are Scalable Vision Learners <https://arxiv.org/abs/2111.06377>"

    Only some of the patches pass through the encoder and then the decoder tries to reconstruct
    the hidden patches, which improves training speed.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 16,
        masking_ratio: float = 0.75,
        decoder_hidden_size: int = 384,
        decoder_mlp_dim: int = 512,
        decoder_num_layers: int = 4,
        decoder_num_heads: int = 16,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        decoder_pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        classification: bool = False,
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
            num_heads: number of attention heads. Defaults to 16.
            masking_ratio: ratio of patches to be masked. Defaults to 0.75.
            decoder_hidden_size: dimension of hidden layer for decoder. Defaults to 384.
            decoder_mlp_dim: dimension of feedforward layer for decoder. Defaults to 512.
            decoder_num_layers: number of transformer blocks for decoder. Defaults to 4.
            decoder_num_heads: number of attention heads for decoder. Defaults to 16.
            proj_type: position embedding layer type. Defaults to "conv".
            pos_embed_type: position embedding layer type. Defaults to "learnable".
            decoder_pos_embed_type: position embedding layer type for decoder. Defaults to "learnable".
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dimensions. Defaults to 3.
            qkv_bias: apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), and sin-cos positional encoding
            >>> net = MaskedAutoencoderViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16), pos_embed_type='sincos')

            # for 3-channel with image size of (128,128,128) a learnable positional encoding and classification backbone
            >>> net = MaskedAutoencoderViT(in_channels=3, img_size=(128,128,128), patch_size=(16,16,16), classification=True)

            # for 3-channel with image size of (224,224) and a masking ratio of 0.25
            >>> net = MaskedAutoencoderViT(in_channels=3, img_size=(224,224), patch_size=(16,16,16), masking_ratio=0.25)

        """

        super().__init__()
        if not is_sqrt(patch_size):
            raise ValueError(f"patch_size should be square number, got {patch_size}.")
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.spatial_dims = spatial_dims
        for m, p in zip(self.img_size, self.patch_size):
            if m % p != 0:
                raise ValueError(f"patch_size={patch_size} should be divisible by img_size={img_size}.")

        self.classification = classification
        self.decoder_hidden_size = decoder_hidden_size
        if masking_ratio <= 0 or masking_ratio >= 1:
            raise ValueError(f"masking_ratio should be in the range (0, 1), got {masking_ratio}.")

        self.masking_ratio = masking_ratio
        self.proj_type = proj_type
        if self.classification:
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
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        # decoder
        self.decoder_embed = nn.Linear(hidden_size, decoder_hidden_size)

        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

        self.decoder_pos_embed_type = look_up_option(decoder_pos_embed_type, SUPPORTED_POS_EMBEDDING_TYPES)
        self.decoder_pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches, decoder_hidden_size))

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_hidden_size, decoder_mlp_dim, decoder_num_heads, dropout_rate, qkv_bias, save_attn
                )
                for i in range(decoder_num_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_hidden_size)
        self.decoder_pred = nn.Linear(decoder_hidden_size, int(np.prod(patch_size)) * in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        adapted from monai/networks/blocks/patchembedding.py and https://github.com/facebookresearch/mae
        """
        if self.decoder_pos_embed_type == "none":
            pass
        elif self.decoder_pos_embed_type == "learnable":
            trunc_normal_(self.decoder_pos_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.decoder_pos_embed_type == "sincos":
            grid_size = []
            for in_size, pa_size in zip(self.img_size, self.patch_size):
                grid_size.append(in_size // pa_size)

            with torch.no_grad():
                pos_embeddings = build_sincos_position_embedding(grid_size, self.decoder_hidden_size, self.spatial_dims)
                self.decoder_pos_embedding.data.copy_(pos_embeddings.float())
                self.decoder_pos_embedding.requires_grad = False
        else:
            raise ValueError(f"decoder_pos_embed_type {self.decoder_pos_embed_type} not supported.")

        # initialize patch_embedding like nn.Linear (instead of nn.Conv2d)
        if self.proj_type == "conv":
            w = self.patch_embedding.patch_embeddings.weight.data
            torch.nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        trunc_normal_(self.mask_tokens, mean=0.0, std=0.02, a=-2.0, b=2.0)
        if self.classification:
            trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def forward(self, x):
        """
        adapted from https://github.com/facebookresearch/mae
        """
        x = self.patch_embedding(x)
        x, mask, ids_restore = self.random_masking(x)

        if hasattr(self, "cls_token"):
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # decoder
        x = self.decoder_embed(x)

        if hasattr(self, "cls_token"):
            mask_tokens = self.mask_tokens.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = (
                torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                + self.decoder_pos_embedding
            )
            x = torch.cat([x[:, :1, :], x_], dim=1)
        else:
            mask_tokens = self.mask_tokens.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = (
                torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                + self.decoder_pos_embedding
            )  # unshuffle

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        if self.classification:
            return x[:, 1:, :], mask

        else:
            return x, mask

    def random_masking(self, x):
        """
        Random masking of patches in the input tensor x.
        adapted from https://github.com/facebookresearch/mae

        Returns:
            x_masked: masked input tensor
            mask: binary mask
            ids_restore: indices to restore the original order
        """
        batch_size, num_tokens, dimension = x.shape

        noise = torch.rand(batch_size, num_tokens, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        number_of_tokens_to_keep = int(num_tokens * (1 - self.masking_ratio))
        ids_keep = ids_shuffle[:, :number_of_tokens_to_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dimension))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, num_tokens], device=x.device)
        mask[:, :number_of_tokens_to_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
