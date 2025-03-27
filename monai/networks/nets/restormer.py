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

import torch
import torch.nn as nn

from monai.networks.blocks.cablock import CABlock, FeedForward
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.downsample import DownSample
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Norm
from monai.utils.enums import DownsampleMode, UpsampleMode


class MDTATransformerBlock(nn.Module):
    """Basic transformer unit combining MDTA and GDFN with skip connections.
    Unlike standard transformers that use LayerNorm, this block uses Instance Norm
    for better adaptation to image restoration tasks.

    Args:
        spatial_dims: Number of spatial dimensions (2D or 3D)
        dim: Number of input channels
        num_heads: Number of attention heads
        ffn_expansion_factor: Expansion factor for feed-forward network
        bias: Whether to use bias in attention layers
        layer_norm_use_bias: Whether to use bias in layer normalization. Defaults to False.
        flash_attention: Whether to use flash attention optimization. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_use_bias: bool = False,
        flash_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = Norm[Norm.INSTANCE, spatial_dims](dim, affine=layer_norm_use_bias)
        self.attn = CABlock(spatial_dims, dim, num_heads, bias, flash_attention)
        self.norm2 = Norm[Norm.INSTANCE, spatial_dims](dim, affine=layer_norm_use_bias)
        self.ffn = FeedForward(spatial_dims, dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(Convolution):
    """Initial feature extraction using overlapped convolutions.
    Unlike standard patch embeddings that use non-overlapping patches,
    this approach maintains spatial continuity through 3x3 convolutions.

    Args:
        spatial_dims: Number of spatial dimensions (2D or 3D)
        in_channels: Number of input channels
        embed_dim: Dimension of embedded features. Defaults to 48.
        bias: Whether to use bias in convolution layer. Defaults to False.
    """

    def __init__(self, spatial_dims: int, in_channels: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=3,
            strides=1,
            padding=1,
            bias=bias,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x


class Restormer(nn.Module):
    """Restormer: Efficient Transformer for High-Resolution Image Restoration.

    Implements a U-Net style architecture with transformer blocks, combining:
    - Multi-scale feature processing through progressive down/upsampling
    - Efficient attention via MDTA blocks
    - Local feature mixing through GDFN
    - Skip connections for preserving spatial details

    Architecture:
        - Encoder: Progressive feature downsampling with increasing channels
        - Latent: Deep feature processing at lowest resolution
        - Decoder: Progressive upsampling with skip connections
        - Refinement: Final feature enhancement
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple[int, ...] = (1, 1, 1, 1),
        heads: tuple[int, ...] = (1, 1, 1, 1),
        num_refinement_blocks: int = 4,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_use_bias: bool = True,
        dual_pixel_task: bool = False,
        flash_attention: bool = False,
    ) -> None:
        super().__init__()
        """Initialize Restormer model.

        Args:
            spatial_dims: Number of spatial dimensions (2D or 3D)
            in_channels: Number of input image channels
            out_channels: Number of output image channels
            dim: Base feature dimension. Defaults to 48.
            num_blocks: Number of transformer blocks at each scale. Defaults to (1,1,1,1).
            heads: Number of attention heads at each scale. Defaults to (1,1,1,1).
            num_refinement_blocks: Number of final refinement blocks. Defaults to 4.
            ffn_expansion_factor: Expansion factor for feed-forward network. Defaults to 2.66.
            bias: Whether to use bias in convolutions. Defaults to False.
            layer_norm_use_bias: Whether to use bias in layer normalization. Defaults to True.
            dual_pixel_task: Enable dual-pixel specific processing. Defaults to False.
            flash_attention: Use flash attention if available. Defaults to False.

        Note:
            The number of blocks must be greater than 1
            The length of num_blocks and heads must be equal
            All values in num_blocks must be greater than 0
        """
        # Check input parameters
        assert len(num_blocks) > 1, "Number of blocks must be greater than 1"
        assert len(num_blocks) == len(heads), "Number of blocks and heads must be equal"
        assert all(n > 0 for n in num_blocks), "Number of blocks must be greater than 0"

        # Initial feature extraction
        self.patch_embed = OverlapPatchEmbed(spatial_dims, in_channels, dim)
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()
        num_steps = len(num_blocks) - 1
        self.num_steps = num_steps
        self.spatial_dims = spatial_dims
        spatial_multiplier = 2 ** (spatial_dims - 1)

        # Define encoder levels
        for n in range(num_steps):
            current_dim = dim * (2) ** (n)
            next_dim = current_dim // spatial_multiplier
            self.encoder_levels.append(
                nn.Sequential(
                    *[
                        MDTATransformerBlock(
                            spatial_dims=spatial_dims,
                            dim=current_dim,
                            num_heads=heads[n],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            layer_norm_use_bias=layer_norm_use_bias,
                            flash_attention=flash_attention,
                        )
                        for _ in range(num_blocks[n])
                    ]
                )
            )

            self.downsamples.append(
                DownSample(
                    spatial_dims=self.spatial_dims,
                    in_channels=current_dim,
                    out_channels=next_dim,
                    mode=DownsampleMode.PIXELUNSHUFFLE,
                    scale_factor=2,
                    bias=bias,
                )
            )

        # Define latent space
        latent_dim = dim * (2) ** (num_steps)
        self.latent = nn.Sequential(
            *[
                MDTATransformerBlock(
                    spatial_dims=spatial_dims,
                    dim=latent_dim,
                    num_heads=heads[num_steps],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_use_bias=layer_norm_use_bias,
                    flash_attention=flash_attention,
                )
                for _ in range(num_blocks[num_steps])
            ]
        )

        # Define decoder levels
        for n in reversed(range(num_steps)):
            current_dim = dim * (2) ** (n)
            next_dim = dim * (2) ** (n + 1)
            self.upsamples.append(
                UpSample(
                    spatial_dims=self.spatial_dims,
                    in_channels=next_dim,
                    out_channels=(current_dim),
                    mode=UpsampleMode.PIXELSHUFFLE,
                    scale_factor=2,
                    bias=bias,
                    apply_pad_pool=False,
                )
            )

            # Reduce channel layers to deal with skip connections
            if n != 0:
                self.reduce_channels.append(
                    Convolution(
                        spatial_dims=self.spatial_dims,
                        in_channels=next_dim,
                        out_channels=current_dim,
                        kernel_size=1,
                        bias=bias,
                        conv_only=True,
                    )
                )
                decoder_dim = current_dim
            else:
                decoder_dim = next_dim

            self.decoder_levels.append(
                nn.Sequential(
                    *[
                        MDTATransformerBlock(
                            spatial_dims=spatial_dims,
                            dim=decoder_dim,
                            num_heads=heads[n],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            layer_norm_use_bias=layer_norm_use_bias,
                            flash_attention=flash_attention,
                        )
                        for _ in range(num_blocks[n])
                    ]
                )
            )

        # Final refinement and output
        self.refinement = nn.Sequential(
            *[
                MDTATransformerBlock(
                    spatial_dims=spatial_dims,
                    dim=decoder_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_use_bias=layer_norm_use_bias,
                    flash_attention=flash_attention,
                )
                for _ in range(num_refinement_blocks)
            ]
        )
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=dim,
                out_channels=dim * 2,
                kernel_size=1,
                bias=bias,
                conv_only=True,
            )
        self.output = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=dim * 2,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            bias=bias,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Restormer.
        Processes input through encoder-decoder architecture with skip connections.
        Args:
            inp_img: Input image tensor of shape (B, C, H, W, [D])

        Returns:
            Restored image tensor of shape (B, C, H, W, [D])
        """
        assert all(
            x.shape[-i] > 2**self.num_steps for i in range(1, self.spatial_dims + 1)
        ), "All spatial dimensions should be larger than 2^number_of_step"

        # Patch embedding
        x = self.patch_embed(x)
        skip_connections = []

        # Encoding path
        for _idx, (encoder, downsample) in enumerate(zip(self.encoder_levels, self.downsamples)):
            x = encoder(x)
            skip_connections.append(x)
            x = downsample(x)

        # Latent space
        x = self.latent(x)

        # Decoding path
        for idx in range(len(self.decoder_levels)):
            x = self.upsamples[idx](x)
            x = torch.concat([x, skip_connections[-(idx + 1)]], 1)
            if idx < len(self.decoder_levels) - 1:
                x = self.reduce_channels[idx](x)
            x = self.decoder_levels[idx](x)

        # Final refinement
        x = self.refinement(x)

        if self.dual_pixel_task:
            x = x + self.skip_conv(skip_connections[0])
            x = self.output(x)
        else:
            x = self.output(x)

        return x
