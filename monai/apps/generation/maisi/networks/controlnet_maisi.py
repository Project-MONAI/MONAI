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

import torch
from torch import nn

from monai.networks.blocks import Convolution
from monai.networks.nets.controlnet import ControlNet, ControlNetConditioningEmbedding, zero_module
from monai.networks.nets.diffusion_model_unet import get_down_block, get_mid_block, get_timestep_embedding
from monai.utils import ensure_tuple_rep


class ControlNetMaisi(ControlNet):
    """
    Control network for diffusion models based on Zhang and Agrawala "Adding Conditional Control to Text-to-Image
    Diffusion Models" (https://arxiv.org/abs/2302.05543)

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
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
        conditioning_embedding_in_channels: number of input channels for the conditioning embedding.
        conditioning_embedding_num_channels: number of channels for the blocks in the conditioning embedding.
        use_checkpointing: if True, use activation checkpointing to save memory.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        include_modality_input: if True, use modality input.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
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
        conditioning_embedding_in_channels: int = 1,
        conditioning_embedding_num_channels: Sequence[int] = (16, 32, 96, 256),
        use_checkpointing: bool = True,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        include_modality_input: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "ControlNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "to be specified when with_conditioning=True."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError("ControlNet expects with_conditioning=True when specifying the cross_attention_dim.")

        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError(
                f"ControlNet expects all channels to be a multiple of norm_num_groups, but got"
                f" channels={num_channels} and norm_num_groups={norm_num_groups}"
            )

        if len(num_channels) != len(attention_levels):
            raise ValueError(
                f"ControlNet expects channels to have the same length as attention_levels, but got "
                f"channels={num_channels} and attention_levels={attention_levels}"
            )

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                f"num_head_channels should have the same length as attention_levels, but got channels={num_channels} "
                f"and attention_levels={attention_levels} . For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                f"`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                f"`num_channels`, but got num_res_blocks={num_res_blocks} and channels={num_channels}."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning
        self.use_checkpointing = use_checkpointing
        self.include_modality_input = include_modality_input

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        time_embed_dim = num_channels[0] * 4
        self.time_embed = self._create_embedding_module(num_channels[0], time_embed_dim)

        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        new_time_embed_dim = time_embed_dim
        if self.include_modality_input:
            self.modality_layer = self._create_embedding_module(1, time_embed_dim)
            new_time_embed_dim += time_embed_dim

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            spatial_dims=spatial_dims,
            in_channels=conditioning_embedding_in_channels,
            channels=conditioning_embedding_num_channels,
            out_channels=num_channels[0],
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]

        controlnet_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=output_channel,
            out_channels=output_channel,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        controlnet_block = zero_module(controlnet_block.conv)
        self.controlnet_down_blocks.append(controlnet_block)

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
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )
            self.down_blocks.append(down_block)

            for _ in range(num_res_blocks[i]):
                controlnet_block = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=output_channel,
                    out_channels=output_channel,
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
            if not is_final_block:
                controlnet_block = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=output_channel,
                    out_channels=output_channel,
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        mid_block_channel = num_channels[-1]
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=mid_block_channel,
            temb_channels=new_time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        controlnet_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=output_channel,
            out_channels=output_channel,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.controlnet_mid_block = zero_module(controlnet_block)

    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        return model

    def _validate_input_tensor(self, tensor, tensor_name, include_flag_name, expected_last_dim, emb):
        if tensor is None:
            raise ValueError(f"{tensor_name} should be provided when {include_flag_name} is True.")
        if tensor.dim() != 2 or tensor.shape[1] != expected_last_dim:
            raise ValueError(f"{tensor_name} should have shape (N, {expected_last_dim}), got {tuple(tensor.shape)}.")
        return tensor.to(dtype=emb.dtype)

    def _get_input_embeddings(self, emb, modality):
        if self.include_modality_input:
            modality = self._validate_input_tensor(modality, "modality_tensor", "include_modality_input", 1, emb)
            _emb = self.modality_layer(modality)
            emb = torch.cat((emb, _emb), dim=1)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        modality_tensor: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        emb = self._prepare_time_and_class_embedding(x, timesteps, class_labels, modality_tensor)
        h = self._apply_initial_convolution(x)
        if self.use_checkpointing:
            controlnet_cond = torch.utils.checkpoint.checkpoint(
                self.controlnet_cond_embedding, controlnet_cond, use_reentrant=False
            )
        else:
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        h += controlnet_cond
        down_block_res_samples, h = self._apply_down_blocks(emb, context, h)
        h = self._apply_mid_block(emb, context, h)
        down_block_res_samples, mid_block_res_sample = self._apply_controlnet_blocks(h, down_block_res_samples)
        # scaling
        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]
        mid_block_res_sample *= conditioning_scale

        return down_block_res_samples, mid_block_res_sample

    def _prepare_time_and_class_embedding(self, x, timesteps, class_labels, modality_tensor):
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

        emb = self._get_input_embeddings(emb, modality_tensor)
        return emb

    def _apply_initial_convolution(self, x):
        # 3. initial convolution
        h = self.conv_in(x)
        return h

    def _apply_down_blocks(self, emb, context, h):
        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        return down_block_res_samples, h

    def _apply_mid_block(self, emb, context, h):
        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)
        return h

    def _apply_controlnet_blocks(self, h, down_block_res_samples):
        # 6. Control net blocks
        controlnet_down_block_res_samples = []
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples.append(down_block_res_sample)

        mid_block_res_sample = self.controlnet_mid_block(h)

        return controlnet_down_block_res_samples, mid_block_res_sample
