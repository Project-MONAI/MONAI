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

from typing import TYPE_CHECKING, Sequence, cast

import torch

from monai.utils import optional_import

ControlNet, has_controlnet = optional_import("generative.networks.nets.controlnet", name="ControlNet")
get_timestep_embedding, has_get_timestep_embedding = optional_import(
    "generative.networks.nets.diffusion_model_unet", name="get_timestep_embedding"
)

if TYPE_CHECKING:
    from generative.networks.nets.controlnet import ControlNet as ControlNetType
else:
    ControlNetType = cast(type, ControlNet)


class ControlNetMaisi(ControlNetType):
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
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        conditioning_embedding_in_channels: number of input channels for the conditioning embedding.
        conditioning_embedding_num_channels: number of channels for the blocks in the conditioning embedding.
        use_checkpointing: if True, use activation checkpointing to save memory.
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
        use_flash_attention: bool = False,
        conditioning_embedding_in_channels: int = 1,
        conditioning_embedding_num_channels: Sequence[int] | None = (16, 32, 96, 256),
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            num_res_blocks,
            num_channels,
            attention_levels,
            norm_num_groups,
            norm_eps,
            resblock_updown,
            num_head_channels,
            with_conditioning,
            transformer_num_layers,
            cross_attention_dim,
            num_class_embeds,
            upcast_attention,
            use_flash_attention,
            conditioning_embedding_in_channels,
            conditioning_embedding_num_channels,
        )
        self.use_checkpointing = use_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        emb = self._prepare_time_and_class_embedding(x, timesteps, class_labels)
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

    def _prepare_time_and_class_embedding(self, x, timesteps, class_labels):
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
