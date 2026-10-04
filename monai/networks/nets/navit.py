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

from collections.abc import Callable, Sequence
from functools import partial
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from monai.networks.blocks.mlp import MLPBlock
from monai.utils import ensure_tuple_rep, optional_import

rearrange, has_einops = optional_import("einops", name="rearrange")
repeat, _ = optional_import("einops", name="repeat")

__all__ = ["NaViT"]


def _group_images_by_max_seq_len(
    images: list[Tensor], patch_size: int, calc_token_dropout: Callable | None = None, max_seq_len: int = 2048
) -> list[list[Tensor]]:
    """Group a flat list of variable-size images into packed batches that each fit within ``max_seq_len`` tokens.

    Args:
        images: flat list of image tensors, each of shape ``(C, *spatial)``.
        patch_size: isotropic patch size used to compute the number of tokens per image.
        calc_token_dropout: optional callable ``(*spatial_dims) -> float`` returning the fraction of tokens
            to drop for an image of the given spatial size. If ``None``, no dropout is assumed.
        max_seq_len: maximum number of tokens allowed per packed group.

    Returns:
        List of groups, where each group is a list of image tensors that together fit within ``max_seq_len``.
    """
    groups: list[list[Tensor]] = []
    group: list[Tensor] = []
    seq_len = 0

    for image in images:
        assert isinstance(image, Tensor)
        spatial_dims = image.shape[1:]
        num_patches = 1
        for d in spatial_dims:
            num_patches *= d // patch_size

        image_seq_len = num_patches
        if calc_token_dropout is not None:
            image_seq_len = max(1, int(image_seq_len * (1.0 - calc_token_dropout(*spatial_dims))))

        if image_seq_len > max_seq_len:
            raise ValueError(
                f"Image with spatial dimensions {spatial_dims} produces {image_seq_len} tokens, "
                f"which exceeds max_seq_len={max_seq_len}."
            )

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if group:
        groups.append(group)

    return groups


class _RMSNorm(nn.Module):
    """Per-head RMS normalization applied to query and key tensors.

    Equivalent to the QK-norm introduced in ViT-22B
    (Dehghani et al., https://arxiv.org/abs/2302.05442).

    Args:
        num_heads: number of attention heads.
        dim_head: dimension of each head.
    """

    def __init__(self, num_heads: int, dim_head: int) -> None:
        super().__init__()
        self.scale = dim_head**0.5
        self.gamma = nn.Parameter(torch.ones(num_heads, 1, dim_head))

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, F.normalize(x, dim=-1) * self.scale * self.gamma)


class _NaViTAttention(nn.Module):
    """Multi-head attention with QK-normalization and support for packed-sequence attention masks.

    This block is used both for the main transformer layers (self-attention) and for the final
    attention-pooling step (cross-attention between learned queries and patch tokens).

    Args:
        hidden_size: dimension of the token embeddings.
        num_heads: number of attention heads.
        dim_head: dimension of each head. Defaults to ``hidden_size // num_heads``.
        dropout_rate: dropout probability applied to attention weights and output projection.
        qkv_bias: whether to add a bias term to the QKV linear projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dim_head: int | None = None,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.num_heads = num_heads
        self.dim_head = dim_head if dim_head is not None else hidden_size // num_heads
        inner_dim = self.num_heads * self.dim_head

        self.norm = nn.LayerNorm(hidden_size)
        self.q_norm = _RMSNorm(num_heads, self.dim_head)
        self.k_norm = _RMSNorm(num_heads, self.dim_head)

        self.to_q = nn.Linear(hidden_size, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(inner_dim, hidden_size, bias=False)

        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output = nn.Dropout(dropout_rate)
        self.scale = self.dim_head**-0.5

    def forward(self, x: Tensor, context: Tensor | None = None, attn_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: query tensor of shape ``(B, N, C)``.
            context: key/value source tensor of shape ``(B, M, C)``. When ``None``, self-attention is performed.
            attn_mask: boolean mask of shape ``(B, 1, N, M)`` where ``True`` indicates positions that
                **should** be attended to. Positions with ``False`` are masked out (set to ``-inf``).

        Returns:
            Tensor of shape ``(B, N, C)``.
        """
        x = self.norm(x)
        kv_src = context if context is not None else x

        # project and reshape to (B, heads, seq, dim_head)
        q = self.to_q(x).unflatten(-1, (self.num_heads, self.dim_head)).transpose(1, 2)
        k = self.to_k(kv_src).unflatten(-1, (self.num_heads, self.dim_head)).transpose(1, 2)
        v = self.to_v(kv_src).unflatten(-1, (self.num_heads, self.dim_head)).transpose(1, 2)

        # QK normalization for training stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attn_mask is not None:
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.drop_weights(dots.softmax(dim=-1))
        out = torch.matmul(attn, v)  # (B, heads, N, dim_head)
        out = out.transpose(1, 2).flatten(-2)  # (B, N, inner_dim)
        return cast(Tensor, self.drop_output(self.out_proj(out)))


class _NaViTTransformerBlock(nn.Module):
    """Single NaViT transformer block: pre-norm self-attention followed by pre-norm MLP.

    Args:
        hidden_size: token embedding dimension.
        mlp_dim: hidden dimension of the feed-forward network.
        num_heads: number of attention heads.
        dim_head: per-head dimension. Defaults to ``hidden_size // num_heads``.
        dropout_rate: dropout probability.
        qkv_bias: whether to add bias to QKV projections.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dim_head: int | None = None,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn = _NaViTAttention(hidden_size, num_heads, dim_head, dropout_rate, qkv_bias)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: input tensor of shape ``(B, N, C)``.
            attn_mask: packed-sequence attention mask of shape ``(B, 1, N, N)``.

        Returns:
            Tensor of shape ``(B, N, C)``.
        """
        x = self.attn(x, attn_mask=attn_mask) + x
        x = self.mlp(self.norm(x)) + x
        return x


class NaViT(nn.Module):
    """NaViT: Native Resolution Vision Transformer with Patch n' Pack, extended to 2D and 3D.

    Based on: "Dehghani et al., Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution
    <https://arxiv.org/abs/2307.06304>"

    NaViT removes the fixed-resolution constraint of standard ViT by packing multiple variable-size images
    into a single sequence. Key features:

    - **Patch n' Pack**: multiple images (possibly different resolutions) are concatenated into one sequence
      per batch element, separated by a per-image attention mask.
    - **Factorized positional embeddings**: separate learnable embeddings for each spatial axis are summed,
      allowing generalization to unseen resolutions.
    - **Token dropout**: a configurable fraction of patch tokens can be randomly dropped during training,
      acting as a form of masked-image modelling.
    - **Attention pooling**: a learned query vector attends over each image's tokens to produce a fixed-size
      per-image representation, cleanly handling variable numbers of images per packed sequence.
    - **QK normalization**: RMS normalization on queries and keys for training stability (ViT-22B).
    - **spatial_dims support**: works for 2D images ``(C, H, W)`` and 3D volumes ``(C, H, W, D)``.

    Args:
        image_size (Union[Sequence[int], int]): reference spatial size used to derive the positional embedding
            tables. Each axis gets a table of size ``image_size[i] // patch_size``. Images at inference time
            may differ from this size as long as each dimension is divisible by ``patch_size``.
        patch_size (int): isotropic patch size. All spatial dimensions of every input image must be divisible
            by this value.
        num_classes (int): number of output classes for the classification head.
        hidden_size (int): token embedding dimension.
        mlp_dim (int): hidden dimension of the feed-forward network inside each transformer block.
        num_layers (int): number of transformer blocks.
        num_heads (int): number of attention heads.
        in_channels (int): number of input image channels. Defaults to 1 (grayscale medical images).
        dim_head (int, optional): per-head dimension. Defaults to ``hidden_size // num_heads``.
        dropout_rate (float): dropout probability applied inside transformer blocks. Defaults to 0.0.
        emb_dropout_rate (float): dropout probability applied to patch embeddings. Defaults to 0.0.
        token_dropout_prob (Union[float, Callable, None]): fraction of patch tokens to randomly drop during
            training. Accepts a float in ``(0, 1)``, or a callable ``(*spatial_dims) -> float`` for
            resolution-dependent dropout. ``None`` disables token dropout. Defaults to ``None``.
        spatial_dims (int): number of spatial dimensions, either 2 or 3. Defaults to 3.
        qkv_bias (bool): whether to add bias to QKV projections. Defaults to False.

    Raises:
        ValueError: When ``spatial_dims`` is not 2 or 3.
        ValueError: When ``dropout_rate`` or ``emb_dropout_rate`` is outside ``[0, 1]``.
        ValueError: When ``hidden_size`` is not divisible by ``num_heads``.
        ValueError: When ``token_dropout_prob`` is a float outside ``(0, 1)``.

    Examples::

        # 3D single-channel (e.g. CT) classification with variable-resolution volumes
        >>> net = NaViT(
        ...     image_size=96, patch_size=16, num_classes=2,
        ...     hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
        ...     in_channels=1, spatial_dims=3,
        ... )
        >>> volumes = [
        ...     [torch.randn(1, 96, 96, 96), torch.randn(1, 64, 64, 64)],
        ...     [torch.randn(1, 80, 96, 80)],
        ... ]
        >>> logits = net(volumes)  # shape: (3, 2)

        # 2D RGB classification (e.g. pathology patches) with token dropout
        >>> net2d = NaViT(
        ...     image_size=256, patch_size=32, num_classes=10,
        ...     hidden_size=512, mlp_dim=2048, num_layers=6, num_heads=8,
        ...     in_channels=3, spatial_dims=2, token_dropout_prob=0.1,
        ... )
        >>> images = [
        ...     [torch.randn(3, 256, 256), torch.randn(3, 128, 128)],
        ...     [torch.randn(3, 192, 256)],
        ... ]
        >>> logits2d = net2d(images)  # shape: (3, 10)
    """

    def __init__(
        self,
        image_size: Sequence[int] | int,
        patch_size: int,
        num_classes: int,
        hidden_size: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        in_channels: int = 1,
        dim_head: int | None = None,
        dropout_rate: float = 0.0,
        emb_dropout_rate: float = 0.0,
        token_dropout_prob: float | Callable | None = None,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if not (0 <= emb_dropout_rate <= 1):
            raise ValueError("emb_dropout_rate should be between 0 and 1.")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.in_channels = in_channels

        # --- token dropout ---
        self.calc_token_dropout: Callable | None = None
        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob
        elif isinstance(token_dropout_prob, (float, int)):
            if not (0.0 < float(token_dropout_prob) < 1.0):
                raise ValueError("token_dropout_prob must be in (0, 1) when given as a float.")
            _prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda *_dims: _prob

        # --- patch embedding ---
        # patch_dim = channels * patch_size^spatial_dims
        patch_dim = in_channels * (patch_size**spatial_dims)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, hidden_size), nn.LayerNorm(hidden_size)
        )

        # --- factorized positional embeddings (one table per spatial axis) ---
        image_size_t = ensure_tuple_rep(image_size, spatial_dims)
        for i, img_d in enumerate(image_size_t):
            if img_d % patch_size != 0:
                raise ValueError(f"image_size dimension {i} ({img_d}) must be divisible by patch_size ({patch_size}).")
        self.pos_embed_axes = nn.ParameterList(
            [nn.Parameter(torch.randn(img_d // patch_size, hidden_size)) for img_d in image_size_t]
        )

        self.emb_dropout = nn.Dropout(emb_dropout_rate)

        # --- transformer ---
        self.blocks = nn.ModuleList(
            [
                _NaViTTransformerBlock(hidden_size, mlp_dim, num_heads, dim_head, dropout_rate, qkv_bias)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        # --- attention pooling ---
        self.attn_pool_query = nn.Parameter(torch.randn(hidden_size))
        self.attn_pool = _NaViTAttention(hidden_size, num_heads, dim_head, dropout_rate, qkv_bias)

        # --- classification head ---
        self.mlp_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, num_classes, bias=False))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights following standard ViT practice."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for param in self.pos_embed_axes:
            nn.init.trunc_normal_(param, std=0.02)
        nn.init.trunc_normal_(self.attn_pool_query, std=0.02)

    @property
    def device(self) -> torch.device:
        """Return the device on which the model parameters reside."""
        return next(self.parameters()).device

    def _image_to_patches(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Rearrange a single image into a sequence of flattened patch tokens and their grid positions.

        Args:
            image: tensor of shape ``(C, *spatial)`` where ``spatial`` has ``self.spatial_dims`` dimensions.

        Returns:
            seq: patch token tensor of shape ``(num_patches, patch_dim)``.
            pos: integer grid-coordinate tensor of shape ``(num_patches, spatial_dims)``.
        """
        p = self.patch_size
        spatial = image.shape[1:]  # (H, W) or (H, W, D)

        if self.spatial_dims == 2:
            h, w = spatial
            # (C, H, W) -> (H/p * W/p, C * p * p)
            seq = image.unfold(1, p, p).unfold(2, p, p)  # (C, H/p, W/p, p, p)
            seq = seq.permute(1, 2, 0, 3, 4).reshape(-1, self.in_channels * p * p)
            gh, gw = h // p, w // p
            pos_h = torch.arange(gh, device=image.device)
            pos_w = torch.arange(gw, device=image.device)
            grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing="ij"), dim=-1)  # (gh, gw, 2)
            pos = grid.reshape(-1, 2)
        else:
            h, w, d = spatial
            # (C, H, W, D) -> (H/p * W/p * D/p, C * p * p * p)
            seq = image.unfold(1, p, p).unfold(2, p, p).unfold(3, p, p)  # (C, H/p, W/p, D/p, p, p, p)
            seq = seq.permute(1, 2, 3, 0, 4, 5, 6).reshape(-1, self.in_channels * p * p * p)
            gh, gw, gd = h // p, w // p, d // p
            pos_h = torch.arange(gh, device=image.device)
            pos_w = torch.arange(gw, device=image.device)
            pos_d = torch.arange(gd, device=image.device)
            grid = torch.stack(torch.meshgrid(pos_h, pos_w, pos_d, indexing="ij"), dim=-1)  # (gh, gw, gd, 3)
            pos = grid.reshape(-1, 3)

        return seq, pos

    def forward(
        self, batched_images: list[list[Tensor]], group_images: bool = False, group_max_seq_len: int = 2048
    ) -> Tensor:
        """Run NaViT on a batch of packed image groups.

        Args:
            batched_images: a list of groups, where each group is a list of image tensors.
                Each image must have shape ``(C, *spatial)`` with ``C == in_channels`` and every
                spatial dimension divisible by ``patch_size``.  The outer list corresponds to the
                batch dimension; images within the same group are packed into a single sequence.
                If ``group_images=True``, a flat ``list[Tensor]`` may be passed instead and the
                packing is performed automatically.
            group_images: when ``True``, treat ``batched_images`` as a flat list of tensors and
                automatically pack them into groups of at most ``group_max_seq_len`` tokens.
            group_max_seq_len: maximum sequence length used when ``group_images=True``.

        Returns:
            Tensor of shape ``(total_images, num_classes)`` containing one logit vector per image
            across all groups in the batch.

        Raises:
            ValueError: If an image does not have the expected number of dimensions
                (``spatial_dims + 1``).
            ValueError: If an image's channel count does not match ``in_channels``.
            ValueError: If any spatial dimension of an image is not divisible by ``patch_size``.
        """
        device = self.device
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        # optional auto-packing
        if group_images:
            batched_images = _group_images_by_max_seq_len(
                batched_images,  # type: ignore[arg-type]
                patch_size=self.patch_size,
                calc_token_dropout=self.calc_token_dropout,
                max_seq_len=group_max_seq_len,
            )

        # ------------------------------------------------------------------ #
        # 1. Convert each image to patch tokens + grid positions              #
        # ------------------------------------------------------------------ #
        num_images_per_group: list[int] = []
        batched_sequences: list[Tensor] = []
        batched_positions: list[Tensor] = []
        batched_image_ids: list[Tensor] = []

        for images in batched_images:
            num_images_per_group.append(len(images))
            sequences: list[Tensor] = []
            positions: list[Tensor] = []
            image_ids = torch.empty((0,), device=device, dtype=torch.long)

            for image_id, image in enumerate(images):
                if image.ndim != self.spatial_dims + 1:
                    raise ValueError(
                        f"Expected image with {self.spatial_dims + 1} dimensions (C, *spatial), "
                        f"got shape {tuple(image.shape)}."
                    )
                if image.shape[0] != self.in_channels:
                    raise ValueError(f"Expected {self.in_channels} input channels, got {image.shape[0]}.")
                spatial = image.shape[1:]
                for dim_size in spatial:
                    if dim_size % self.patch_size != 0:
                        raise ValueError(
                            f"All spatial dimensions must be divisible by patch_size={self.patch_size}, "
                            f"got spatial shape {spatial}."
                        )

                seq, pos = self._image_to_patches(image)  # (N, patch_dim), (N, spatial_dims)

                # optional token dropout (training only)
                if self.calc_token_dropout is not None and self.training:
                    dropout_frac = self.calc_token_dropout(*spatial)
                    num_keep = max(1, int(seq.shape[0] * (1.0 - dropout_frac)))
                    keep_idx = torch.randn(seq.shape[0], device=device).topk(num_keep).indices
                    seq = seq[keep_idx]
                    pos = pos[keep_idx]

                image_ids = F.pad(image_ids, (0, seq.shape[0]), value=image_id)
                sequences.append(seq)
                positions.append(pos)

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))

        # ------------------------------------------------------------------ #
        # 2. Pad sequences to the same length and build attention masks       #
        # ------------------------------------------------------------------ #
        lengths = torch.tensor([s.shape[0] for s in batched_sequences], device=device, dtype=torch.long)
        max_len = int(lengths.amax().item())
        len_range = torch.arange(max_len, device=device)

        # key-padding mask: True for valid (non-padded) positions
        key_pad_mask = len_range.unsqueeze(0) < lengths.unsqueeze(1)  # (B, max_len)

        # per-image attention mask: tokens from different images must not attend to each other
        batched_image_ids_padded = pad_sequence(batched_image_ids)  # (B, max_len)
        same_image = batched_image_ids_padded.unsqueeze(2) == batched_image_ids_padded.unsqueeze(
            1
        )  # (B, max_len, max_len)
        attn_mask = same_image & key_pad_mask.unsqueeze(1)  # (B, max_len, max_len)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, max_len, max_len)

        # ------------------------------------------------------------------ #
        # 3. Patch embedding + factorized positional encoding                 ##
        # ------------------------------------------------------------------ #
        patches = pad_sequence(batched_sequences)  # (B, max_len, patch_dim)
        patch_positions = pad_sequence(batched_positions)  # (B, max_len, spatial_dims)

        x = self.to_patch_embedding(patches)  # (B, max_len, hidden_size)

        # sum positional embeddings from each axis
        for axis_idx, pos_embed in enumerate(self.pos_embed_axes):
            axis_indices = patch_positions[..., axis_idx]  # (B, max_len)
            # clamp to handle positions beyond the reference image_size table
            axis_indices = axis_indices.clamp(max=pos_embed.shape[0] - 1)
            x = x + pos_embed[axis_indices]

        x = self.emb_dropout(x)

        # ------------------------------------------------------------------ #
        # 4. Transformer                                                       #
        # ------------------------------------------------------------------ #
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.norm(x)

        # ------------------------------------------------------------------ #
        # 5. Attention pooling: one query per image in the group              #
        # ------------------------------------------------------------------ #
        num_images_t = torch.tensor(num_images_per_group, device=device, dtype=torch.long)
        max_queries = int(num_images_t.amax().item())

        # expand the shared query vector to (B, max_queries, hidden_size)
        queries = self.attn_pool_query.unsqueeze(0).unsqueeze(0).expand(x.shape[0], max_queries, -1)

        # build cross-attention mask: query i attends only to tokens belonging to image i
        image_id_range = torch.arange(max_queries, device=device)
        pool_mask = image_id_range.unsqueeze(1) == batched_image_ids_padded.unsqueeze(1)  # (B, max_queries, max_len)
        pool_mask = pool_mask & key_pad_mask.unsqueeze(1)  # (B, max_queries, max_len)
        pool_mask = pool_mask.unsqueeze(1)  # (B, 1, max_queries, max_len)

        pooled = self.attn_pool(queries, context=x, attn_mask=pool_mask) + queries  # (B, max_queries, hidden_size)

        # ------------------------------------------------------------------ #
        # 6. Flatten, filter padding queries, and classify                    #
        # ------------------------------------------------------------------ #
        pooled = pooled.reshape(-1, pooled.shape[-1])  # (B * max_queries, hidden_size)

        is_valid = (image_id_range.unsqueeze(0) < num_images_t.unsqueeze(1)).reshape(-1)  # (B * max_queries,)
        pooled = pooled[is_valid]  # (total_images, hidden_size)

        return cast(Tensor, self.mlp_head(pooled))  # (total_images, num_classes)
