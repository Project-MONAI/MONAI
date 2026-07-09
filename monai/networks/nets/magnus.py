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

"""
MAGNUS: Multi-Attention Guided Network for Unified Segmentation via CNN-ViT Fusion

A hybrid CNN-Transformer architecture that combines multi-scale CNN features
with Vision Transformer representations through cross-modal attention fusion
for advanced medical image segmentation.

Reference:
    Aras, E., Kayikcioglu, T., Aras, S., & Merd, N. (2026).
    MAGNUS: Multi-Attention Guided Network for Unified Segmentation via CNN-ViT Fusion.
    IEEE Access. DOI: 10.1109/ACCESS.2026.3656667
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.utils import get_act_layer, get_norm_layer

__all__ = [
    "MAGNUS",
    "CNNPath",
    "TransformerPath",
    "CrossModalAttentionFusion",
    "ScaleAdaptiveConv",
    "MagnusSEBlock",
    "DecoderBlock",
]


class CNNPath(nn.Module):
    """
    CNN encoder path with strided convolutions for hierarchical feature extraction.

    Args:
        spatial_dims: number of spatial dimensions (2 or 3).
        in_channels: number of input channels.
        features: sequence of output channels for each encoder stage.
        norm: feature normalization type, one of ("batch", "instance", "group").
        act: activation type, one of ("relu", "leakyrelu", "prelu", "gelu").
        dropout: dropout ratio after each convolution block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        features: Sequence[int],
        norm: str | tuple = "batch",
        act: str | tuple = "relu",
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the CNN encoder path.

        See class docstring for argument descriptions.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.stages = nn.ModuleList()
        current_channels = in_channels

        for feat in features:
            stage = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=current_channels,
                    out_channels=feat,
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    norm=norm,
                    act=act,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=feat,
                    out_channels=feat,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    norm=norm,
                    act=act,
                    dropout=dropout,
                ),
            )
            self.stages.append(stage)
            current_channels = feat

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass returning features from each stage.

        Args:
            x: input tensor of shape ``(B, C, *spatial_dims)``.

        Returns:
            List of feature tensors from each encoder stage,
            ordered from shallow to deep.
        """
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class TransformerPath(nn.Module):
    """
    Vision Transformer path for global context modeling.

    Applies patch embedding followed by transformer encoder layers
    to capture long-range dependencies. Includes learnable positional
    embeddings that are interpolated to match varying input sizes.

    Args:
        spatial_dims: number of spatial dimensions (2 or 3).
        in_channels: number of input channels.
        hidden_dim: transformer hidden dimension.
        num_heads: number of attention heads.
        depth: number of transformer encoder layers.
        patch_size: size of patches for embedding.
        dropout: dropout rate in transformer layers.
        mlp_ratio: ratio of mlp hidden dim to embedding dim.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        patch_size: int = 16,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        """
        Initialize the Vision Transformer path.

        See class docstring for argument descriptions.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Patch embedding via convolution
        conv_type = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.embedding = conv_type(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable positional embedding (will be interpolated for different input sizes)
        # Initialize with a reasonable default size, will adapt dynamically
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Dropout for positional embedding
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def _interpolate_pos_encoding(self, x: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Interpolate positional embeddings to match the number of patches.

        Args:
            x: input tensor for device reference.
            num_patches: target number of patches.

        Returns:
            Interpolated positional embeddings of shape (1, num_patches, hidden_dim).
        """
        if num_patches == self.pos_embed.shape[1]:
            return self.pos_embed

        # Interpolate positional embeddings
        pos_embed = self.pos_embed.transpose(1, 2)  # (1, hidden_dim, N)
        pos_embed = F.interpolate(pos_embed, size=num_patches, mode="linear", align_corners=False)
        result: torch.Tensor = pos_embed.transpose(1, 2)  # (1, num_patches, hidden_dim)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer path.

        Args:
            x: input tensor of shape ``(B, C, *spatial_dims)``.

        Returns:
            Transformed features of shape ``(B, hidden_dim, *reduced_spatial_dims)``.
        """
        # Patch embedding: (B, C, D, H, W) -> (B, hidden_dim, Dp, Hp, Wp)
        x_embedded = self.embedding(x)
        batch_size = x_embedded.shape[0]
        spatial_shape = x_embedded.shape[2:]

        # Flatten spatial dims: (B, hidden_dim, *spatial) -> (B, N, hidden_dim)
        x_flat = x_embedded.flatten(2).transpose(1, 2)
        num_patches = x_flat.shape[1]

        # Add positional encoding
        pos_embed = self._interpolate_pos_encoding(x_flat, num_patches)
        x_flat = x_flat + pos_embed
        x_flat = self.pos_drop(x_flat)

        # Apply transformer
        x_transformed = self.transformer(x_flat)
        x_transformed = self.norm(x_transformed)

        # Reshape back to spatial: (B, N, hidden_dim) -> (B, hidden_dim, *spatial)
        x_out: torch.Tensor = x_transformed.transpose(1, 2).view(batch_size, self.hidden_dim, *spatial_shape)

        return x_out


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion between CNN and Transformer features.

    Performs bidirectional cross-attention where CNN features attend to
    Transformer features and vice versa, then combines the results.

    Args:
        spatial_dims: number of spatial dimensions (2 or 3).
        channels: number of input/output channels.
        num_heads: number of attention heads.
        dropout: dropout rate for attention weights.
    """

    def __init__(self, spatial_dims: int, channels: int, num_heads: int, dropout: float = 0.0) -> None:
        """
        Initialize the cross-modal attention fusion module.

        See class docstring for argument descriptions.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")

        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)

        conv_type = nn.Conv3d if spatial_dims == 3 else nn.Conv2d

        # QKV projections for both paths
        self.to_qkv_cnn = conv_type(channels, channels * 3, 1, bias=False)
        self.to_qkv_vit = conv_type(channels, channels * 3, 1, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            conv_type(channels, channels, 1), nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cross-modal attention fusion.

        Args:
            cnn_feat: CNN features of shape ``(B, C, *spatial_dims)``.
            vit_feat: ViT features of shape ``(B, C, *spatial_dims_vit)``.

        Returns:
            Fused features of shape ``(B, C, *spatial_dims)``.
        """
        batch_size, channels = cnn_feat.shape[:2]
        spatial_shape = cnn_feat.shape[2:]
        heads = self.num_heads

        # Interpolate ViT features to match CNN spatial dimensions
        if cnn_feat.shape[2:] != vit_feat.shape[2:]:
            mode = "trilinear" if self.spatial_dims == 3 else "bilinear"
            vit_feat = F.interpolate(vit_feat, size=spatial_shape, mode=mode, align_corners=False)

        # Compute Q, K, V for both paths
        q_c, k_c, v_c = self.to_qkv_cnn(cnn_feat).chunk(3, dim=1)
        q_v, k_v, v_v = self.to_qkv_vit(vit_feat).chunk(3, dim=1)

        # Reshape for multi-head attention: (B, heads, head_dim, N)
        def reshape_for_attention(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, heads, self.head_dim, -1)

        q_c, k_c, v_c = map(reshape_for_attention, (q_c, k_c, v_c))
        q_v, k_v, v_v = map(reshape_for_attention, (q_v, k_v, v_v))

        # Cross-attention: CNN queries attend to ViT keys/values
        attn_cv = torch.einsum("b h d i, b h d j -> b h i j", q_c, k_v) * self.scale
        attn_cv = self.dropout(attn_cv.softmax(dim=-1))
        out_c = torch.einsum("b h i j, b h d j -> b h d i", attn_cv, v_v)

        # Cross-attention: ViT queries attend to CNN keys/values
        attn_vc = torch.einsum("b h d i, b h d j -> b h i j", q_v, k_c) * self.scale
        attn_vc = self.dropout(attn_vc.softmax(dim=-1))
        out_v = torch.einsum("b h i j, b h d j -> b h d i", attn_vc, v_c)

        # Reshape back to spatial
        out_c = out_c.contiguous().view(batch_size, channels, *spatial_shape)
        out_v = out_v.contiguous().view(batch_size, channels, *spatial_shape)

        # Combine and project
        fused: torch.Tensor = self.to_out(out_c + out_v)

        return fused


class ScaleAdaptiveConv(nn.Module):
    """
    Scale-adaptive convolution module with multiple kernel sizes.

    Applies parallel convolutions with different kernel sizes and
    combines the outputs for multi-scale feature extraction.

    Args:
        spatial_dims: number of spatial dimensions (2 or 3).
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_sizes: sequence of kernel sizes to use.
        norm: normalization type.
        act: activation type.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        norm: str | tuple = "batch",
        act: str | tuple = "relu",
    ) -> None:
        """
        Initialize the scale-adaptive convolution module.

        See class docstring for argument descriptions.
        """
        super().__init__()
        self.spatial_dims = spatial_dims

        conv_type = nn.Conv3d if spatial_dims == 3 else nn.Conv2d

        self.convs = nn.ModuleList(
            [conv_type(in_channels, out_channels, k, padding=k // 2, bias=False) for k in kernel_sizes]
        )

        # Shared normalization and activation
        self.norm = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)
        self.act = get_act_layer(name=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale convolutions.

        Args:
            x: input tensor of shape ``(B, C, *spatial_dims)``.

        Returns:
            Multi-scale features of shape ``(B, out_channels, *spatial_dims)``.
        """
        outs = [conv(x) for conv in self.convs]
        out = torch.stack(outs, dim=0).sum(dim=0)
        out = self.norm(out)
        result: torch.Tensor = self.act(out)
        return result


class MagnusSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel recalibration in MAGNUS.

    Args:
        spatial_dims: number of spatial dimensions (2 or 3).
        channels: number of input/output channels.
        reduction: channel reduction ratio for the squeeze operation.
    """

    def __init__(self, spatial_dims: int, channels: int, reduction: int = 16) -> None:
        """
        Initialize the Squeeze-and-Excitation block.

        See class docstring for argument descriptions.
        """
        super().__init__()
        self.spatial_dims = spatial_dims

        pool_type = nn.AdaptiveAvgPool3d if spatial_dims == 3 else nn.AdaptiveAvgPool2d
        self.avg_pool = pool_type(1)

        reduced_channels = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SE block.

        Args:
            x: input tensor of shape ``(B, C, *spatial_dims)``.

        Returns:
            Channel-recalibrated tensor of same shape.
        """
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)

        # Reshape for broadcasting
        if self.spatial_dims == 3:
            y = y.view(b, c, 1, 1, 1)
        else:
            y = y.view(b, c, 1, 1)

        result: torch.Tensor = x * y.expand_as(x)
        return result


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling, skip connection, and SE attention.

    Args:
        spatial_dims: number of spatial dimensions (2 or 3).
        in_channels: number of input channels.
        skip_channels: number of skip connection channels.
        out_channels: number of output channels.
        norm: normalization type.
        act: activation type.
        dropout: dropout ratio.
        use_se: whether to use SE block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str | tuple = "batch",
        act: str | tuple = "relu",
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        """
        Initialize the decoder block.

        See class docstring for argument descriptions.
        """
        super().__init__()
        self.spatial_dims = spatial_dims

        # Upsampling with UpSample block
        self.upsample = UpSample(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=2,
            mode="nontrainable",
            interp_mode="trilinear" if spatial_dims == 3 else "bilinear",
            align_corners=False,
        )

        # Convolution after concatenation with skip
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            norm=norm,
            act=act,
            dropout=dropout,
        )

        # Optional SE block
        self.se = MagnusSEBlock(spatial_dims, out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for decoder block.

        Args:
            x: input tensor from previous decoder stage.
            skip: skip connection tensor from encoder.

        Returns:
            Decoded features tensor.
        """
        x = self.upsample(x)

        # Handle spatial dimension mismatch
        if x.shape[2:] != skip.shape[2:]:
            mode = "trilinear" if self.spatial_dims == 3 else "bilinear"
            x = F.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)

        return x


class MAGNUS(nn.Module):
    """
    MAGNUS: Multi-scale Attention Guided Network for Unified Segmentation.

    A hybrid CNN-Transformer architecture that combines:
    - CNN path with strided convolutions for hierarchical feature extraction
    - Vision Transformer path for global context modeling
    - Cross-modal attention fusion for enhanced feature representation
    - Scale-adaptive convolutions for multi-scale analysis
    - Decoder with SE attention and deep supervision support

    Args:
        spatial_dims: number of spatial dimensions (2 or 3).
        in_channels: number of input image channels.
        out_channels: number of output segmentation classes.
        features: sequence of feature channels for encoder stages.
            Default: (64, 128, 256, 512).
        vit_depth: number of transformer encoder layers. Default: 6.
        vit_patch_size: patch size for ViT embedding. Default: 16.
        vit_num_heads: number of attention heads in ViT. If None, computed as
            features[-1] // 32. Default: None.
        fusion_num_heads: number of attention heads in cross-modal fusion.
            If None, uses vit_num_heads. Default: None.
        scale_kernel_sizes: kernel sizes for scale-adaptive conv. Default: (3, 5, 7).
        norm: normalization type ("batch", "instance", "group"). Default: "batch".
        act: activation type. Default: "relu".
        dropout: dropout ratio. Default: 0.0.
        vit_dropout: dropout ratio for transformer. Default: 0.1.
        deep_supervision: whether to return auxiliary outputs. Default: False.
        aux_weights: suggested weights for auxiliary losses when using deep supervision.
            These weights are stored as an attribute for user convenience but are NOT
            applied internally. Users should apply them externally when computing the
            total loss. Default: (0.4, 0.3, 0.3).

    Example:
        >>> import torch
        >>> from monai.networks.nets import MAGNUS
        >>> # 3D segmentation
        >>> model = MAGNUS(spatial_dims=3, in_channels=1, out_channels=2)
        >>> x = torch.randn(1, 1, 64, 64, 64)
        >>> y = model(x)
        >>> print(y.shape)  # torch.Size([1, 2, 64, 64, 64])
        >>> # 2D segmentation
        >>> model_2d = MAGNUS(spatial_dims=2, in_channels=3, out_channels=4)
        >>> x_2d = torch.randn(1, 3, 256, 256)
        >>> y_2d = model_2d(x_2d)
        >>> print(y_2d.shape)  # torch.Size([1, 4, 256, 256])

    Reference:
        Aras, E., Kayikcioglu, T., Aras, S., & Merd, N. (2026).
        MAGNUS: Multi-Attention Guided Network for Unified Segmentation via CNN-ViT Fusion.
        IEEE Access. DOI: 10.1109/ACCESS.2026.3656667
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        vit_depth: int = 6,
        vit_patch_size: int = 16,
        vit_num_heads: int | None = None,
        fusion_num_heads: int | None = None,
        scale_kernel_sizes: Sequence[int] = (3, 5, 7),
        norm: str | tuple = "batch",
        act: str | tuple = "relu",
        dropout: float = 0.0,
        vit_dropout: float = 0.1,
        deep_supervision: bool = False,
        aux_weights: Sequence[float] = (0.4, 0.3, 0.3),
    ) -> None:
        """
        Initialize the MAGNUS model.

        See class docstring for argument descriptions.
        """
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = list(features)
        self.deep_supervision = deep_supervision
        self.aux_weights = list(aux_weights)

        # Compute number of attention heads
        vit_hidden_dim = self.features[-1]
        if vit_num_heads is None:
            vit_num_heads = max(vit_hidden_dim // 32, 1)
        if fusion_num_heads is None:
            fusion_num_heads = vit_num_heads

        # CNN encoder path
        self.cnn_path = CNNPath(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            features=self.features,
            norm=norm,
            act=act,
            dropout=dropout,
        )

        # Transformer path
        self.transformer_path = TransformerPath(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            hidden_dim=vit_hidden_dim,
            num_heads=vit_num_heads,
            depth=vit_depth,
            patch_size=vit_patch_size,
            dropout=vit_dropout,
        )

        # Cross-modal attention fusion
        self.fusion = CrossModalAttentionFusion(
            spatial_dims=spatial_dims, channels=vit_hidden_dim, num_heads=fusion_num_heads, dropout=dropout
        )

        # Scale-adaptive convolution
        self.scale_conv = ScaleAdaptiveConv(
            spatial_dims=spatial_dims,
            in_channels=vit_hidden_dim,
            out_channels=vit_hidden_dim,
            kernel_sizes=scale_kernel_sizes,
            norm=norm,
            act=act,
        )

        # Decoder path
        reversed_features = list(reversed(self.features))
        self.decoder_blocks = nn.ModuleList()
        self.aux_heads = nn.ModuleList()

        for i in range(len(reversed_features) - 1):
            in_ch = reversed_features[i]
            out_ch = reversed_features[i + 1]

            self.decoder_blocks.append(
                DecoderBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    skip_channels=out_ch,
                    out_channels=out_ch,
                    norm=norm,
                    act=act,
                    dropout=dropout,
                    use_se=True,
                )
            )

            # Auxiliary segmentation heads for deep supervision
            if deep_supervision:
                conv_type = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
                self.aux_heads.append(conv_type(out_ch, out_channels, kernel_size=1))

        # Final segmentation head
        conv_type = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.final_conv = conv_type(self.features[0], out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of MAGNUS.

        Args:
            x: input tensor of shape ``(B, in_channels, *spatial_dims)``.

        Returns:
            If ``deep_supervision`` is False, returns segmentation logits of shape
            ``(B, out_channels, *spatial_dims)``.
            If ``deep_supervision`` is True, returns tuple of (main_output, auxiliary_outputs)
            where auxiliary_outputs is a list of intermediate segmentation maps.
        """
        input_shape = x.shape[2:]

        # 1. CNN feature extraction
        cnn_features = self.cnn_path(x)
        cnn_deepest = cnn_features[-1]

        # 2. Transformer path
        vit_features = self.transformer_path(x)

        # 3. Cross-modal attention fusion
        fused_features = self.fusion(cnn_deepest, vit_features)

        # 4. Scale-adaptive convolution
        scale_features = self.scale_conv(cnn_deepest)

        # 5. Combine fused and scale features
        combined = fused_features + scale_features

        # 6. Decoder with skip connections
        decoder_out = combined
        cnn_skips = list(reversed(cnn_features[:-1]))
        aux_outputs = []

        for i, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, cnn_skips)):
            decoder_out = decoder_block(decoder_out, skip)

            # Auxiliary outputs for deep supervision
            if self.deep_supervision and i < len(self.aux_heads):
                aux_out = self.aux_heads[i](decoder_out)
                aux_out = F.interpolate(
                    aux_out,
                    size=input_shape,
                    mode="trilinear" if self.spatial_dims == 3 else "bilinear",
                    align_corners=False,
                )
                aux_outputs.append(aux_out)

        # 7. Final segmentation
        seg_logits = self.final_conv(decoder_out)

        # Upsample to original input size if needed
        if seg_logits.shape[2:] != input_shape:
            seg_logits = F.interpolate(
                seg_logits,
                size=input_shape,
                mode="trilinear" if self.spatial_dims == 3 else "bilinear",
                align_corners=False,
            )

        if self.deep_supervision:
            return seg_logits, aux_outputs

        result: torch.Tensor = seg_logits
        return result
