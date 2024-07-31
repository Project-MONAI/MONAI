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

from collections.abc import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.type_conversion import convert_to_tensor

get_down_block, has_get_down_block = optional_import(
    "generative.networks.nets.diffusion_model_unet", name="get_down_block"
)
get_mid_block, has_get_mid_block = optional_import(
    "generative.networks.nets.diffusion_model_unet", name="get_mid_block"
)
get_timestep_embedding, has_get_timestep_embedding = optional_import(
    "generative.networks.nets.diffusion_model_unet", name="get_timestep_embedding"
)
get_up_block, has_get_up_block = optional_import("generative.networks.nets.diffusion_model_unet", name="get_up_block")
xformers, has_xformers = optional_import("xformers")
zero_module, has_zero_module = optional_import("generative.networks.nets.diffusion_model_unet", name="zero_module")

__all__ = ["DiffusionModelUNetMaisi"]


class DiffusionModelUNetMaisi(nn.Module):
    """
    U-Net network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: Number of spatial dimensions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks (see ResnetBlock) per level. Can be a single integer or a sequence of integers.
        num_channels: Tuple of block output channels.
        attention_levels: List of levels to add attention.
        norm_num_groups: Number of groups for the normalization.
        norm_eps: Epsilon for the normalization.
        resblock_updown: If True, use residual blocks for up/downsampling.
        num_head_channels: Number of channels in each attention head. Can be a single integer or a sequence of integers.
        with_conditioning: If True, add spatial transformers to perform conditioning.
        transformer_num_layers: Number of layers of Transformer blocks to use.
        cross_attention_dim: Number of context dimensions to use.
        num_class_embeds: If specified (as an int), then this model will be class-conditional with `num_class_embeds` classes.
        upcast_attention: If True, upcast attention operations to full precision.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: If different from zero, this will be the dropout value for the cross-attention layers.
        include_top_region_index_input: If True, use top region index input.
        include_bottom_region_index_input: If True, use bottom region index input.
        include_spacing_input: If True, use spacing input.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
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
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_top_region_index_input: bool = False,
        include_bottom_region_index_input: bool = False,
        include_spacing_input: bool = False,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelUNetMaisi expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNetMaisi expects with_conditioning=True when specifying the cross_attention_dim."
            )
        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError(
                f"DiffusionModelUNetMaisi expects all num_channels being multiple of norm_num_groups, "
                f"but get num_channels: {num_channels} and norm_num_groups: {norm_num_groups}"
            )

        if len(num_channels) != len(attention_levels):
            raise ValueError(
                f"DiffusionModelUNetMaisi expects num_channels being same size of attention_levels, "
                f"but get num_channels: {len(num_channels)} and attention_levels: {len(attention_levels)}"
            )

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention and not has_xformers:
            raise ValueError("use_flash_attention is True but xformers is not installed.")

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = num_channels[0] * 4
        self.time_embed = self._create_embedding_module(num_channels[0], time_embed_dim)

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        self.include_top_region_index_input = include_top_region_index_input
        self.include_bottom_region_index_input = include_bottom_region_index_input
        self.include_spacing_input = include_spacing_input

        new_time_embed_dim = time_embed_dim
        if self.include_top_region_index_input:
            self.top_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_bottom_region_index_input:
            self.bottom_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_spacing_input:
            self.spacing_layer = self._create_embedding_module(3, time_embed_dim)
            new_time_embed_dim += time_embed_dim

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
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
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=new_time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
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
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.up_blocks.append(up_block)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        return model

    def _get_time_and_class_embedding(self, x, timesteps, class_labels):
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb += class_emb
        return emb

    def _get_input_embeddings(self, emb, top_index, bottom_index, spacing):
        if self.include_top_region_index_input:
            _emb = self.top_region_index_layer(top_index)
            emb = torch.cat((emb, _emb), dim=1)
        if self.include_bottom_region_index_input:
            _emb = self.bottom_region_index_layer(bottom_index)
            emb = torch.cat((emb, _emb), dim=1)
        if self.include_spacing_input:
            _emb = self.spacing_layer(spacing)
            emb = torch.cat((emb, _emb), dim=1)
        return emb

    def _apply_down_blocks(self, h, emb, context, down_block_additional_residuals):
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)

        # Additional residual conections for Controlnets
        if down_block_additional_residuals is not None:
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples.append(down_block_res_sample)

            down_block_res_samples = new_down_block_res_samples
        return h, down_block_res_samples

    def _apply_up_blocks(self, h, emb, context, down_block_res_samples):
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)

        return h

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the UNet model.

        Args:
            x: Input tensor of shape (N, C, SpatialDims).
            timesteps: Timestep tensor of shape (N,).
            context: Context tensor of shape (N, 1, ContextDim).
            class_labels: Class labels tensor of shape (N,).
            down_block_additional_residuals: Additional residual tensors for down blocks of shape (N, C, FeatureMapsDims).
            mid_block_additional_residual: Additional residual tensor for mid block of shape (N, C, FeatureMapsDims).
            top_region_index_tensor: Tensor representing top region index of shape (N, 4).
            bottom_region_index_tensor: Tensor representing bottom region index of shape (N, 4).
            spacing_tensor: Tensor representing spacing of shape (N, 3).

        Returns:
            A tensor representing the output of the UNet model.
        """

        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)
        h = self.conv_in(x)
        h, _updated_down_block_res_samples = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        h = self.middle_block(h, emb, context)

        # Additional residual conections for Controlnets
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual

        h = self._apply_up_blocks(h, emb, context, _updated_down_block_res_samples)
        h = self.out(h)
        h_tensor: torch.Tensor = convert_to_tensor(h)
        return h_tensor
