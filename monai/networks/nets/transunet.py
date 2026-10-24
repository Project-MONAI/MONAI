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

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep

__all__ = ["TransUNet"]


class TransUNet(nn.Module):
    """
    TransUNet based on: "Chen et al.,
    TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
    <https://arxiv.org/abs/2102.04306>".

    A hybrid CNN-Transformer segmentation network: a convolutional stem extracts features at
    1/16 of the input resolution, which are tokenized and processed by a stack of transformer
    blocks, and a U-Net-like decoder with cascaded upsampling combines the transformer output
    with the CNN skip connections back to full resolution.

    Unlike the original 2D implementation, the network is dimensionally generic and supports
    2D and 3D inputs via ``spatial_dims``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        norm_name: tuple | str = "instance",
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of the transformer hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            num_layers: number of transformer blocks. Defaults to 12.
            norm_name: feature normalization type and arguments. Defaults to "instance".
            res_block: if residual blocks are used in the CNN stem and decoder. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 2.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.

        Examples::

            # for single channel input 4-channel output with image size of (96,96)
            >>> net = TransUNet(in_channels=1, out_channels=4, img_size=(96, 96))

            # for 3-channel input 3-channel output with image size of (128,128,128)
            >>> net = TransUNet(in_channels=3, out_channels=3, img_size=(128, 128, 128), spatial_dims=3)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.stem_downsampling = 16
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        stem_stride = ensure_tuple_rep(self.stem_downsampling, spatial_dims)
        self.feat_size = tuple(img_d // s_d for img_d, s_d in zip(img_size, stem_stride))
        if any(img_d % s_d for img_d, s_d in zip(img_size, stem_stride)):
            raise ValueError("img_size should be divisible by 16.")
        self.hidden_size = hidden_size

        # CNN stem: skips at 1/1, 1/2, 1/4 and 1/8 scale, bottleneck features at 1/16 scale
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size * 4,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 16,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # transformer encoder: each spatial position of the 1/16-scale feature map is a token
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=feature_size * 16,
            img_size=self.feat_size,
            patch_size=1,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type="conv",
            pos_embed_type="learnable",
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        # decoder: cascaded upsampling with the CNN skip connections
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        x = self.patch_embedding(enc5)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        dec4 = self.proj_feat(x)
        dec3 = self.decoder4(dec4, enc4)
        dec2 = self.decoder3(dec3, enc3)
        dec1 = self.decoder2(dec2, enc2)
        out = self.decoder1(dec1, enc1)
        return self.out(out)
