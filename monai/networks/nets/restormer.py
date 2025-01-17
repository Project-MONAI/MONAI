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


from monai.networks.blocks.upsample import UpSample, UpsampleMode
from monai.networks.blocks.downsample import DownSample, DownsampleMode
from monai.networks.layers.factories import Norm
from monai.networks.blocks.cablock import FeedForward, CABlock
from monai.networks.blocks.convolutions import Convolution


class MDTATransformerBlock(nn.Module):
    """Basic transformer unit combining MDTA and GDFN with skip connections.
    Unlike standard transformers that use LayerNorm, this block uses Instance Norm
    for better adaptation to image restoration tasks."""
    
    def __init__(self, spatial_dims: int, dim: int, num_heads: int, ffn_expansion_factor: float,
                 bias: bool, LayerNorm_type: str, flash_attention: bool = False):
        super().__init__()
        use_bias = LayerNorm_type != 'BiasFree'        
        self.norm1 = Norm[Norm.INSTANCE, 2](dim, affine=use_bias)
        self.attn = CABlock(spatial_dims, dim, num_heads, bias, flash_attention)
        self.norm2 = Norm[Norm.INSTANCE, 2](dim, affine=use_bias)
        self.ffn = FeedForward(spatial_dims, dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Initial feature extraction using overlapped convolutions.
    Unlike standard patch embeddings that use non-overlapping patches,
    this approach maintains spatial continuity through 3x3 convolutions."""
    
    def __init__(self, spatial_dims: int, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_c,
            out_channels=embed_dim,
            kernel_size=3,
            strides=1,
            padding=1,
            bias=bias,
            conv_only=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Restormer_new(nn.Module):
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
    def __init__(self, 
                 spatial_dims=2,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 1, 1, 1],  
                 heads=[1, 1, 1, 1],  
                 num_refinement_blocks=4,    
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 flash_attention=False):
        super().__init__()
        """Initialize Restormer model.
        
        Args:
            inp_channels: Number of input image channels
            out_channels: Number of output image channels
            dim: Base feature dimension
            num_blocks: Number of transformer blocks at each scale
            num_refinement_blocks: Number of final refinement blocks
            heads: Number of attention heads at each scale
            ffn_expansion_factor: Expansion factor for feed-forward network
            bias: Whether to use bias in convolutions
            LayerNorm_type: Type of normalization ('WithBias' or 'BiasFree')
            dual_pixel_task: Enable dual-pixel specific processing
            flash_attention: Use flash attention if available
        """
        # Check input parameters
        assert len(num_blocks) > 1, "Number of blocks must be greater than 1"
        assert len(num_blocks) == len(heads), "Number of blocks and heads must be equal"
        assert all([n > 0 for n in num_blocks]), "Number of blocks must be greater than 0"
        
        # Initial feature extraction
        self.patch_embed = OverlapPatchEmbed(spatial_dims, inp_channels, dim)
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()
        num_steps = len(num_blocks) - 1 
        self.num_steps = num_steps
        self.spatial_dims=spatial_dims

        # Define encoder levels
        for n in range(num_steps):
            current_dim = dim * 2**n
            self.encoder_levels.append(
                nn.Sequential(*[
                    MDTATransformerBlock(
                        spatial_dims=spatial_dims,
                        dim=current_dim,
                        num_heads=heads[n],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        flash_attention=flash_attention
                    ) for _ in range(num_blocks[n])
                ])
            )

            self.downsamples.append(
                DownSample(
                spatial_dims=self.spatial_dims,
                in_channels=current_dim,
                out_channels=current_dim//2,
                mode=DownsampleMode.PIXELUNSHUFFLE,
                scale_factor=2,
                bias=bias,
                )
            )

        # Define latent space
        latent_dim = dim * 2**num_steps
        self.latent = nn.Sequential(*[
            MDTATransformerBlock(
                spatial_dims=spatial_dims,
                dim=latent_dim,
                num_heads=heads[num_steps],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                flash_attention=flash_attention
            ) for _ in range(num_blocks[num_steps])
        ])

        # Define decoder levels
        for n in reversed(range(num_steps)):
            current_dim = dim * 2**n
            next_dim = dim * 2**(n+1)
            self.upsamples.append(
                UpSample(
                spatial_dims=self.spatial_dims,
                in_channels=next_dim,
                out_channels=(next_dim//2),
                mode=UpsampleMode.PIXELSHUFFLE,
                scale_factor=2,
                bias=bias,
                apply_pad_pool=False
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
                        conv_only=True
                    )
                )
                decoder_dim = current_dim
            else:
                decoder_dim = next_dim
            
            self.decoder_levels.append(
                nn.Sequential(*[
                    MDTATransformerBlock(
                        spatial_dims=spatial_dims,
                        dim=decoder_dim,
                        num_heads=heads[n],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        flash_attention=flash_attention
                    ) for _ in range(num_blocks[n])
                ])
            )

        # Final refinement and output
        self.refinement = nn.Sequential(*[
            MDTATransformerBlock(
                spatial_dims=spatial_dims,
                dim=decoder_dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                flash_attention=flash_attention
            ) for _ in range(num_refinement_blocks)
        ])
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=dim,
                out_channels=int(dim*2**1),
                kernel_size=1,
                bias=bias,
                conv_only=True
            )
            
        self.output = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=int(dim*2**1),
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            bias=bias,
            conv_only=True
        )

    def forward(self, x):
        """Forward pass of Restormer.
        Processes input through encoder-decoder architecture with skip connections.
        Args:
            inp_img: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Restored image tensor of shape (B, C, H, W)
        """
        assert x.shape[-1] > 2 ** self.num_steps and x.shape[-2] > 2 ** self.num_steps, "Input dimensions should be larger than 2^number_of_step"

        # Patch embedding
        x = self.patch_embed(x)
        skip_connections = []

        # Encoding path
        for idx, (encoder, downsample) in enumerate(zip(self.encoder_levels, self.downsamples)):
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
