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

# Portions of this code are derived from the original repository at:
# https://github.com/MIC-DKFZ/dynamic-network-architectures
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from monai.networks.blocks.eva_block import EVABlock
from monai.networks.blocks.primus_block import PrimusPatchDecode, PrimusPatchEmbed
from monai.networks.blocks.rope import SpatialRotaryEmbedding
from monai.networks.layers.weight_init import trunc_normal_
from monai.utils import ensure_tuple_rep

__all__ = ["Primus", "create_primus", "convert_primus_state_dict", "PrimusS", "PrimusB", "PrimusM", "PrimusL"]

_PRIMUS_VARIANTS: dict[str, dict[str, Any]] = {
    "S": {"embed_dim": 396, "num_layers": 12, "num_heads": 6},
    "B": {"embed_dim": 792, "num_layers": 12, "num_heads": 12},
    "M": {"embed_dim": 864, "num_layers": 16, "num_heads": 12},
    "L": {"embed_dim": 1056, "num_layers": 24, "num_heads": 16},
}


class Primus(nn.Module):
    """
    Primus transformer segmentation network, following the PrimusV3 design of
    `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation <https://arxiv.org/abs/2503.01835>`_.

    A residual convolutional stem embeds the input into a token grid downsampled by ``2 ** len(depth_per_level)``
    per axis, an EVA-02 style transformer (rotary and absolute position embeddings, SwiGLU MLP, LayerScale)
    processes the tokens, and a light transposed-convolution decoder restores the input resolution.

    Setting ``patch_drop_rate > 0`` randomly drops tokens during training (e.g. for masked-image-modeling
    pre-training); dropped tokens are replaced by a mask token before decoding. Use ``return_mask=True`` in
    :py:meth:`forward` to also get the mask of kept voxels.

    State dicts of ``PrimusV3`` models from ``dynamic-network-architectures`` (as trained by nnU-Net) can be
    converted with :py:meth:`load_old_state_dict`.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        img_size: spatial size of the input. The transformer's position embeddings are built for this size, so
            the network only accepts inputs of this size. Each entry must be divisible by the patch size
            ``2 ** len(depth_per_level)``.
        spatial_dims: number of spatial dimensions.
        embed_dim: token dimension.
        num_layers: number of transformer blocks.
        num_heads: number of attention heads. ``embed_dim // num_heads`` must be divisible by ``2 * spatial_dims``
            when ``use_rope`` is True.
        depth_per_level: number of residual blocks at each downsampling level of the convolutional stem.
        channels_per_level: channels of the stem followed by those of each level, of length
            ``len(depth_per_level) + 1``.
        add_skips: whether the stem adds projected intermediate features to the tokens.
        num_register_tokens: number of learnable register tokens prepended to the sequence.
        use_rope: whether to use rotary position embeddings.
        use_abs_pos_embed: whether to use learnable absolute position embeddings.
        mlp_ratio: ratio of the transformer MLP hidden dimension to ``embed_dim``.
        drop_path_rate: maximum stochastic depth rate, increasing linearly over the blocks.
        patch_drop_rate: fraction of tokens dropped during training.
        dropout_rate: dropout rate of the transformer projections.
        attention_dropout_rate: dropout rate of the attention weights.
        init_values: initial LayerScale value, ``None`` disables LayerScale.
        scale_attn_inner: whether to apply a LayerNorm to the attention output.
        stem_norm: feature normalization of the convolutional stem.
        stem_act: activation of the convolutional stem.
        decoder_act: activation of the decoder.

    Example::

        # PrimusV3-M for 1-channel 128^3 patches, 3 output classes
        net = create_primus("M", in_channels=1, out_channels=3, img_size=128)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        spatial_dims: int = 3,
        embed_dim: int = 864,
        num_layers: int = 16,
        num_heads: int = 12,
        depth_per_level: Sequence[int] = (1, 1, 1),
        channels_per_level: Sequence[int] = (32, 64, 256, 1024),
        add_skips: bool = True,
        num_register_tokens: int = 0,
        use_rope: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio: float = 4 * 2 / 3,
        drop_path_rate: float = 0.2,
        patch_drop_rate: float = 0.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        init_values: float | None = 0.1,
        scale_attn_inner: bool = True,
        stem_norm: tuple | str = ("instance", {"affine": True}),
        stem_act: tuple | str = ("leakyrelu", {"inplace": True}),
        decoder_act: tuple | str = "gelu",
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
        if len(depth_per_level) == 0:
            raise ValueError("depth_per_level must have at least one level.")
        if num_register_tokens < 0:
            raise ValueError(f"num_register_tokens must be non-negative, got {num_register_tokens}.")
        for name, rate in (
            ("drop_path_rate", drop_path_rate),
            ("dropout_rate", dropout_rate),
            ("attention_dropout_rate", attention_dropout_rate),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {rate}.")
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = (2 ** len(depth_per_level),) * spatial_dims
        if any(s % p != 0 for s, p in zip(self.img_size, self.patch_size)):
            raise ValueError(f"img_size {self.img_size} must be divisible by the patch size {self.patch_size}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        if not 0.0 <= patch_drop_rate < 1.0:
            raise ValueError(f"patch_drop_rate must be in [0, 1), got {patch_drop_rate}.")
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = math.prod(self.grid_size)
        self.num_register_tokens = num_register_tokens
        self.patch_drop_rate = patch_drop_rate

        self.down_projection = PrimusPatchEmbed(
            spatial_dims,
            in_channels,
            embed_dim,
            depth_per_level=depth_per_level,
            channels_per_level=channels_per_level,
            add_skips=add_skips,
            norm=stem_norm,
            act=stem_act,
        )
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens > 0 else None
        )
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, self.num_patches + num_register_tokens, embed_dim))
            if use_abs_pos_embed
            else None
        )
        self.rope = SpatialRotaryEmbedding(embed_dim // num_heads, self.grid_size) if use_rope else None
        self.blocks = nn.ModuleList(
            EVABlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                scale_attn_inner=scale_attn_inner,
                num_prefix_tokens=num_register_tokens,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                drop_path_rate=drop_path_rate * i / max(num_layers - 1, 1),
                init_values=init_values,
            )
            for i in range(num_layers)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token: torch.Tensor
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))
        self.up_projection = PrimusPatchDecode(spatial_dims, self.patch_size, embed_dim, out_channels, act=decoder_act)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize as upstream: truncated normal transformer weights, rescaled by depth, He-normal convolutions."""
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            for layer_id, block in enumerate(self.blocks, start=1):
                block = cast(EVABlock, block)
                block.attn.proj.weight.div_(math.sqrt(2.0 * layer_id))
                block.mlp.fc2.weight.div_(math.sqrt(2.0 * layer_id))
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        for module in (self.down_projection, self.up_projection):
            for m in module.modules():
                if isinstance(m, _ConvNd):
                    nn.init.kaiming_normal_(m.weight, a=1e-2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _drop_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Randomly keep a subset of the patch tokens of ``x`` (B, N, C), returning the kept indices."""
        if not self.training or self.patch_drop_rate == 0.0:
            return x, None
        b, n, c = x.shape
        num_keep = max(1, int(n * (1.0 - self.patch_drop_rate)))
        keep_indices = torch.argsort(torch.randn(b, n, device=x.device), dim=-1)[:, :num_keep]
        return x.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, c)), keep_indices

    def _restore_patches(self, x: torch.Tensor, keep_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter the kept tokens back into a full sequence filled with the mask token."""
        b, _, c = x.shape
        full = self.mask_token.to(x.dtype).expand(b, self.num_patches, c).clone()
        full.scatter_(1, keep_indices.unsqueeze(-1).expand(-1, -1, c), x)
        kept = torch.zeros(b, self.num_patches, dtype=torch.bool, device=x.device)
        kept.scatter_(1, keep_indices, True)
        return full, kept

    def forward(
        self, x: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: input of shape ``(B, in_channels, *img_size)``.
            return_mask: whether to also return the boolean mask of shape ``(B, 1, *img_size)`` that is True
                on voxels whose token was kept, or ``None`` when no tokens were dropped.
        """
        if tuple(x.shape[2:]) != self.img_size:
            raise ValueError(f"expected input spatial size {self.img_size}, got {tuple(x.shape[2:])}.")
        x = self.down_projection(x)
        b, c = x.shape[:2]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        if self.register_tokens is not None:
            x = torch.cat([self.register_tokens.expand(b, -1, -1), x], dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        rope = self.rope() if self.rope is not None else None
        registers, patches = x[:, : self.num_register_tokens], x[:, self.num_register_tokens :]
        patches, keep_indices = self._drop_patches(patches)
        if keep_indices is not None:
            x = torch.cat([registers, patches], dim=1)
            if rope is not None:
                rope = rope[keep_indices].unsqueeze(1)  # (B, 1, num_keep, 2 * head_dim)

        for block in self.blocks:
            x = block(x, rope=rope)
        x = self.norm(x)[:, self.num_register_tokens :]

        mask = None
        if keep_indices is not None:
            x, kept = self._restore_patches(x, keep_indices)
            # upsample the token mask to voxels: (B, g0, 1, g1, 1, ...) -> (B, g0, p0, g1, p1, ...)
            interleaved = [n for g, p in zip(self.grid_size, self.patch_size) for n in (g, p)]
            mask = kept.view(b, *[n for g in self.grid_size for n in (g, 1)]).expand(b, *interleaved)
            mask = mask.reshape(b, 1, *self.img_size)
        x = x.transpose(1, 2).reshape(b, c, *self.grid_size)
        out = self.up_projection(x)
        return (out, mask) if return_mask else out

    def load_old_state_dict(self, old_state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load a state dict of a ``PrimusV3`` model from ``dynamic-network-architectures``.

        Args:
            old_state_dict: the state dict to convert and load.
        """
        self.load_state_dict(convert_primus_state_dict(old_state_dict))


def convert_primus_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert a ``PrimusV3`` state dict from ``dynamic-network-architectures`` to the layout of :py:class:`Primus`.
    """
    rules = [
        (r"^eva\.", ""),
        (r"stem\.blocks\.0\.", "stem."),
        (r"stages\.(\d+)\.blocks\.(\d+)\.", r"stages.\1.\2."),
        (r"conv([12])\.conv\.", r"conv\1."),
        (r"conv([12])\.norm\.", r"norm\1."),
        (r"skip\.\d+\.(conv|norm)\.", r"skip.\1."),
    ]
    out = {}
    for key, value in state_dict.items():
        if ".all_modules." in key:  # aliases of the conv/norm parameters
            continue
        for pattern, repl in rules:
            key = re.sub(pattern, repl, key)
        out[key] = value
    return out


def create_primus(variant: str, **kwargs) -> Primus:
    """
    Create a PrimusV3 variant with the transformer configuration of the Primus paper.

    Args:
        variant: one of ``"S"``, ``"B"``, ``"M"`` or ``"L"``.
        kwargs: other arguments of :py:class:`Primus`, e.g. ``in_channels``, ``out_channels`` and ``img_size``.
    """
    config = _PRIMUS_VARIANTS.get(variant.upper())
    if config is None:
        raise ValueError(f"invalid Primus variant {variant}, expected one of {list(_PRIMUS_VARIANTS)}.")
    return Primus(**config, **kwargs)


PrimusS = partial(create_primus, "S")
PrimusB = partial(create_primus, "B")
PrimusM = partial(create_primus, "M")
PrimusL = partial(create_primus, "L")
