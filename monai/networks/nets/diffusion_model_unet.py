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
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution, CrossAttentionBlock, MLPBlock, SABlock, SpatialAttentionBlock, Upsample
from monai.networks.layers.factories import Pool
from monai.utils import ensure_tuple_rep, optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

__all__ = ["DiffusionModelUNet"]


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class DiffusionUNetTransformerBlock(nn.Module):
    """
    A Transformer block that allows for the input dimension to differ from the hidden dimension.

    Args:
        num_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        dropout: dropout probability to use.
        cross_attention_dim: size of the context vector for cross attention.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.

    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = SABlock(
            hidden_size=num_attention_heads * num_head_channels,
            hidden_input_size=num_channels,
            num_heads=num_attention_heads,
            dim_head=num_head_channels,
            dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.ff = MLPBlock(hidden_size=num_channels, mlp_dim=num_channels * 4, act="GEGLU", dropout_rate=dropout)
        self.attn2 = CrossAttentionBlock(
            hidden_size=num_attention_heads * num_head_channels,
            num_heads=num_attention_heads,
            hidden_input_size=num_channels,
            context_input_size=cross_attention_dim,
            dim_head=num_head_channels,
            dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
            use_flash_attention=use_flash_attention,
        )
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn1(self.norm1(x)) + x

        # 2. Cross-Attention
        x = self.attn2(self.norm2(x), context=context) + x

        # 3. Feed-forward
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        num_layers: number of layers of Transformer blocks to use.
        dropout: dropout probability to use.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = num_attention_heads * num_head_channels

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)

        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                DiffusionUNetTransformerBlock(
                    num_channels=inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        inner_dim = x.shape[1]

        if self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        if self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1).reshape(batch, height * width * depth, inner_dim)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.spatial_dims == 2:
            x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        if self.spatial_dims == 3:
            x = x.reshape(batch, height, width, depth, inner_dim).permute(0, 4, 1, 2, 3).contiguous()

        x = self.proj_out(x)
        return x + residual


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class DiffusionUnetDownsample(nn.Module):
    """
    Downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels.
        use_conv: if True uses Convolution instead of Pool average to perform downsampling. In case that use_conv is
            False, the number of output channels must be the same as the number of input channels.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

    def __init__(
        self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        output: torch.Tensor = self.op(x)
        return output


class WrappedUpsample(Upsample):
    """
    Wraps MONAI upsample block to allow for calling with timestep embeddings.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        upsampled: torch.Tensor = super().forward(x)
        return upsampled


class DiffusionUNetResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding  channels.
        out_channels: number of output channels.
        up: if True, performs upsampling.
        down: if True, performs downsampling.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = WrappedUpsample(
                spatial_dims=spatial_dims,
                mode="nontrainable",
                in_channels=in_channels,
                out_channels=in_channels,
                interp_mode="nearest",
                scale_factor=2.0,
                align_corners=None,
            )
        elif down:
            self.downsample = DiffusionUnetDownsample(spatial_dims, in_channels, use_conv=False)

        self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.skip_connection: nn.Module
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        else:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        output: torch.Tensor = self.skip_connection(x) + h
        return output


class DownBlock(nn.Module):
    """
    Unet's down block containing resnet and downsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler: nn.Module | None
            if resblock_updown:
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding  channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsampler: nn.Module | None
        if add_downsample:
            if resblock_updown:
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states).contiguous()
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and cross-attention blocks.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    dropout=dropout_cattn,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsampler: nn.Module | None
        if add_downsample:
            if resblock_updown:
                self.downsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context).contiguous()
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialAttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        del context
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states).contiguous()
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class CrossAttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and cross-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class UpBlock(nn.Module):
    """
    Unet's up block containing resnet and upsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )

        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class AttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:

                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states).contiguous()

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class CrossAttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    dropout=dropout_cattn,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionUNetResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:

                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = WrappedUpsample(
                    spatial_dims=spatial_dims,
                    mode="nontrainable",
                    in_channels=out_channels,
                    out_channels=out_channels,
                    interp_mode="nearest",
                    scale_factor=2.0,
                    post_conv=post_conv,
                    align_corners=None,
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


def get_down_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_downsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    dropout_cattn: float = 0.0,
    include_fc: bool = True,
    use_combined_linear: bool = False,
    use_flash_attention: bool = False,
) -> nn.Module:
    if with_attn:
        return AttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    else:
        return DownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
        )


def get_mid_block(
    spatial_dims: int,
    in_channels: int,
    temb_channels: int,
    norm_num_groups: int,
    norm_eps: float,
    with_conditioning: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    dropout_cattn: float = 0.0,
    include_fc: bool = True,
    use_combined_linear: bool = False,
    use_flash_attention: bool = False,
) -> nn.Module:
    if with_conditioning:
        return CrossAttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    else:
        return AttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )


def get_up_block(
    spatial_dims: int,
    in_channels: int,
    prev_output_channel: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_upsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    dropout_cattn: float = 0.0,
    include_fc: bool = True,
    use_combined_linear: bool = False,
    use_flash_attention: bool = False,
) -> nn.Module:
    if with_attn:
        return AttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
    else:
        return UpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
        )


class DiffusionModelUNet(nn.Module):
    """
    Unet network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResnetBlock) per level.
        channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
            classes.
        upcast_attention: if True, upcast attention operations to full precision.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to True.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelUNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNet expects with_conditioning=True when specifying the cross_attention_dim."
            )
        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("DiffusionModelUNet expects all num_channels being multiple of norm_num_groups")

        if len(channels) != len(attention_levels):
            raise ValueError("DiffusionModelUNet expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))

        if len(num_res_blocks) != len(channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        self.in_channels = in_channels
        self.block_out_channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                dropout_cattn=dropout_cattn,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout_cattn=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(channels) - 1)]

            is_final_block = i == len(channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                dropout_cattn=dropout_cattn,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.up_blocks.append(up_block)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
            down_block_additional_residuals: additional residual tensors for down blocks (N, C, FeatureMapsDims).
            mid_block_additional_residual: additional residual tensor for mid block (N, C, FeatureMapsDims).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)

        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # Additional residual conections for Controlnets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += [down_block_res_sample]

            down_block_res_samples = new_down_block_res_samples

        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)

        # Additional residual conections for Controlnets
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        # 6. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)

        # 7. output block
        output: torch.Tensor = self.out(h)

        return output

    def load_old_state_dict(self, old_state_dict: dict, verbose=False) -> None:
        """
        Load a state dict from a DiffusionModelUNet trained with
        [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels).

        Args:
            old_state_dict: state dict from the old DecoderOnlyTransformer  model.
        """

        new_state_dict = self.state_dict()
        # if all keys match, just load the state dict
        if all(k in new_state_dict for k in old_state_dict):
            print("All keys match, loading state dict.")
            self.load_state_dict(old_state_dict)
            return

        if verbose:
            # print all new_state_dict keys that are not in old_state_dict
            for k in new_state_dict:
                if k not in old_state_dict:
                    print(f"key {k} not found in old state dict")
            # and vice versa
            print("----------------------------------------------")
            for k in old_state_dict:
                if k not in new_state_dict:
                    print(f"key {k} not found in new state dict")

        # copy over all matching keys
        for k in new_state_dict:
            if k in old_state_dict:
                new_state_dict[k] = old_state_dict.pop(k)

        # fix the attention blocks
        attention_blocks = [k.replace(".out_proj.weight", "") for k in new_state_dict if "out_proj.weight" in k]
        for block in attention_blocks:
            # projection
            new_state_dict[f"{block}.out_proj.weight"] = old_state_dict.pop(f"{block}.to_out.0.weight")
            new_state_dict[f"{block}.out_proj.bias"] = old_state_dict.pop(f"{block}.to_out.0.bias")

        # fix the upsample conv blocks which were renamed postconv
        for k in new_state_dict:
            if "postconv" in k:
                old_name = k.replace("postconv", "conv")
                new_state_dict[k] = old_state_dict.pop(old_name)
        if verbose:
            # print all remaining keys in old_state_dict
            print("remaining keys in old_state_dict:", old_state_dict.keys())
        self.load_state_dict(new_state_dict)


class DiffusionModelEncoder(nn.Module):
    """
    Classification Network based on the Encoder of the Diffusion Model, followed by fully connected layers. This network is based on
    Wolleb et al. "Diffusion Models for Medical Anomaly Detection" (https://arxiv.org/abs/2203.04306).

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResnetBlock) per level.
        channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds` classes.
        upcast_attention: if True, upcast attention operations to full precision.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelEncoder expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelEncoder expects with_conditioning=True when specifying the cross_attention_dim."
            )

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("DiffusionModelEncoder expects all num_channels being multiple of norm_num_groups")
        if len(channels) != len(attention_levels):
            raise ValueError("DiffusionModelEncoder expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        self.in_channels = in_channels
        self.block_out_channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = channels[0]
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels)  # - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

        self.out = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, self.out_channels))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)

        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        for downsample_block in self.down_blocks:
            h, _ = downsample_block(hidden_states=h, temb=emb, context=context)

        h = h.reshape(h.shape[0], -1)
        output: torch.Tensor = self.out(h)

        return output
