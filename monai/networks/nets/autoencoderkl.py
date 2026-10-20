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
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, SpatialAttentionBlock, Upsample
from monai.utils import ensure_tuple_rep, optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

__all__ = ["AutoencoderKL"]


def _validate_kernel_stride_parameters(
    kernel_size: int | tuple[int, ...] | None,
    stride: int | tuple[int, ...] | None,
    spatial_dims: int,
    param_name: str = "parameter",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Validate and normalize kernel_size and stride parameters.

    Args:
        kernel_size: int or tuple of ints representing kernel size
        stride: int or tuple of ints representing stride
        spatial_dims: number of spatial dimensions
        param_name: name of parameter for error messages

    Returns:
        Tuple of (normalized_kernel_size, normalized_stride)

    Raises:
        ValueError: if parameters are invalid
    """

    # Check for None values
    if kernel_size is None:
        raise ValueError("kernel_size cannot be None")
    if stride is None:
        raise ValueError("stride cannot be None")

    # Normalize kernel_size to tuple
    if isinstance(kernel_size, int):
        kernel_size_tuple = (kernel_size,) * spatial_dims
    else:
        kernel_size_tuple = tuple(kernel_size)

    # Normalize stride to tuple
    if isinstance(stride, int):
        stride_tuple = (stride,) * spatial_dims
    else:
        stride_tuple = tuple(stride)

    # Validate lengths
    if len(kernel_size_tuple) != spatial_dims:
        raise ValueError(f"{param_name} kernel_size must have length {spatial_dims}, got {len(kernel_size_tuple)}")
    if len(stride_tuple) != spatial_dims:
        raise ValueError(f"{param_name} stride must have length {spatial_dims}, got {len(stride_tuple)}")

    # Validate kernel sizes are odd
    for i, k in enumerate(kernel_size_tuple):
        if k % 2 == 0:
            raise ValueError(f"{param_name} kernel_size at dimension {i} must be odd, got {k}")

    # Validate all values are positive integers
    for i, (k, s) in enumerate(zip(kernel_size_tuple, stride_tuple)):
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"{param_name} kernel_size at dimension {i} must be positive int, got {k}")
        if not isinstance(s, int) or s <= 0:
            raise ValueError(f"{param_name} stride at dimension {i} must be positive int, got {s}")

    return kernel_size_tuple, stride_tuple


def _compute_padding(kernel_size: tuple[int, ...]) -> tuple[int, ...]:
    """
    Compute symmetric padding for odd kernel sizes.

    Padding is derived as:
        padding[d] = kernel_size[d] // 2

    Args:
        kernel_size: Kernel size for each spatial dimension.

    Returns:
        Tuple of padding values for each spatial dimension.
    """
    padding = tuple(k // 2 for k in kernel_size)
    return padding


def _normalize_downsample_parameters(
    downsample_parameters: list[dict] | dict | None,
    num_levels: int,
    spatial_dims: int,
    default_kernel_size: int = 3,
    default_stride: int = 2,
) -> list[dict]:
    """
    Normalize downsampling parameters to canonical internal representation.

    Accepts:
    - None: use defaults for all levels
    - Single dict: apply same params to all levels
    - List of dicts: one dict per level

    Each dict can specify:
    - "kernel_size": int or tuple
    - "stride": int or tuple
    - "padding": int or tuple (auto-computed if omitted)

    Returns:
        List of dicts with normalized keys:
        - Each dict has "kernel_size", "stride", "padding" as tuples
        - Length equals num_levels

    Raises:
        ValueError: if parameters are invalid or inconsistent
    """
    if downsample_parameters is None:
        # Default: use provided defaults for all levels
        default_ks_tuple, default_s_tuple = _validate_kernel_stride_parameters(
            default_kernel_size, default_stride, spatial_dims
        )
        default_padding: tuple[int, ...] = _compute_padding(default_ks_tuple)
        return [
            {"kernel_size": default_ks_tuple, "stride": default_s_tuple, "padding": default_padding}
            for _ in range(num_levels)
        ]

    # If single dict, apply to all levels
    if isinstance(downsample_parameters, dict):
        params_list = [downsample_parameters] * num_levels
    else:
        params_list = list(downsample_parameters)

    # Validate we have the right number of levels
    if len(params_list) != num_levels:
        raise ValueError(f"Expected {num_levels} downsampling parameter dicts (one per level), got {len(params_list)}")

    # Normalize each dict
    normalized = []
    for i, params in enumerate(params_list):
        if not isinstance(params, dict):
            raise ValueError(f"Downsampling parameters at level {i} must be dict, got {type(params)}")

        kernel_size = params.get("kernel_size", default_kernel_size)
        stride = params.get("stride", default_stride)
        padding = params.get("padding", None)

        # Validate and normalize kernel_size and stride
        ks_tuple, s_tuple = _validate_kernel_stride_parameters(kernel_size, stride, spatial_dims, f"Level {i}")

        # Compute padding if not provided
        if padding is None:
            padding_tuple: tuple[int, ...] = _compute_padding(ks_tuple)
        else:
            # Normalize provided padding
            if isinstance(padding, int):
                padding_tuple = (padding,) * spatial_dims
            else:
                padding_tuple = tuple(padding)

            if len(padding_tuple) != spatial_dims:
                raise ValueError(f"Level {i} padding must have length {spatial_dims}, got {len(padding_tuple)}")

        normalized.append({"kernel_size": ks_tuple, "stride": s_tuple, "padding": padding_tuple})

    return normalized


class AsymmetricPad(nn.Module):
    """
    Pad the input tensor asymmetrically along every spatial dimension.

    .. deprecated:: 0.10.0
        This class is deprecated and no longer used by `AEKLDownsample`.
        Use configurable kernel_size and stride parameters instead (see `AEKLDownsample`).

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
    """

    def __init__(self, spatial_dims: int) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply asymmetric padding to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Padded tensor.
        """
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        return x


class _RecordShapeHook(nn.Module):
    """Helper module to record spatial shapes during encoding for decoder restoration."""

    def __init__(self, shape_list: list[tuple[int, ...]]) -> None:
        """
        Args:
            shape_list: List to append shapes to during forward pass.
        """
        super().__init__()
        self.shape_list = shape_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Record spatial dimensions and pass through.

        Args:
            x: Input tensor.

        Returns:
            Input tensor unchanged.
        """
        self.shape_list.append(tuple(x.shape[2:]))
        return x


class _ShapeRestoringUpsample(nn.Module):
    """Upsample to exact target size (recorded by encoder) instead of using scale_factor.

    This handles arbitrary input dimensions (odd, non-power-of-2, anisotropic) by restoring
    to the exact pre-downsampling shape recorded during encoding.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        post_conv: nn.Module,
        shape_index: int,
        downsample_shapes_ref: list,
        scale_factor: tuple[int, ...] | None = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            post_conv: convolution module to apply after upsampling.
            shape_index: index into downsample_shapes_ref list.
            downsample_shapes_ref: reference to shared list of shapes from encoder.
            scale_factor: fallback upsampling scale factor if shape_index is out of range.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.post_conv = post_conv
        self.shape_index = shape_index
        self.downsample_shapes_ref = downsample_shapes_ref  # Reference to the shared list, NOT a module
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample to exact target size, then apply post-convolution."""
        # Get target shape from downsample_shapes (in reverse order)
        if self.downsample_shapes_ref and self.shape_index < len(self.downsample_shapes_ref):
            # Shapes are stored in order, but we're using them in reverse
            target_shape_index = len(self.downsample_shapes_ref) - 1 - self.shape_index
            target_shape = self.downsample_shapes_ref[target_shape_index]
            x = F.interpolate(x, size=target_shape, mode="nearest")
        elif self.scale_factor is not None:
            # Fallback for standalone decode (no encoder run): use scale_factor
            sf = tuple(float(s) for s in self.scale_factor)
            x = F.interpolate(x, scale_factor=sf, mode="nearest")

        x = self.post_conv(x)
        return x


class AEKLDownsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        kernel_size: kernel size for the convolution. Can be int or tuple. Default: 3.
        stride: stride for the convolution. Can be int or tuple. Default: 2.
        padding: padding for the convolution. If None, computed from kernel_size. Default: None.
        use_legacy_padding: if True and padding is None, use asymmetric padding (0,1) for each dimension
            to match the original MONAI Generative implementation. Default: False.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 2,
        padding: int | tuple[int, ...] | None = None,
        use_legacy_padding: bool = False,
    ) -> None:
        super().__init__()

        # Validate and normalize kernel_size and stride
        kernel_size_tuple, stride_tuple = _validate_kernel_stride_parameters(
            kernel_size, stride, spatial_dims, "AEKLDownsample"
        )

        self.use_legacy_padding = use_legacy_padding and (padding is None)
        padding_tuple: tuple[int, ...]
        if self.use_legacy_padding:
            # Legacy behavior: asymmetric padding (0, 1) per dimension + conv with padding=0
            self.pad = (0, 1) * spatial_dims
            padding_tuple = (0,) * spatial_dims
        else:
            # New behavior: compute symmetric padding if not provided
            if padding is None:
                padding_tuple = _compute_padding(kernel_size_tuple)
            else:
                if isinstance(padding, int):
                    padding_tuple = (padding,) * spatial_dims
                else:
                    padding_tuple = tuple(padding)

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=tuple(stride_tuple),
            kernel_size=tuple(kernel_size_tuple),
            padding=tuple(padding_tuple),
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional downsampling.

        Args:
            x: Input tensor.

        Returns:
            Downsampled tensor.
        """
        if self.use_legacy_padding:
            x = F.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class AEKLResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, norm_num_groups: int, norm_eps: float, out_channels: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.nin_shortcut: nn.Module
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        downsample_parameters: list of dicts specifying kernel_size, stride, padding for each downsampling level.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        downsample_parameters: list[dict] | dict | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        # Normalize downsampling parameters
        num_downsample_levels = len(channels) - 1
        use_legacy_padding = downsample_parameters is None  # Track if using legacy defaults
        normalized_downsample_params = _normalize_downsample_parameters(
            downsample_parameters, num_downsample_levels, spatial_dims
        )

        # Store for decoder to use
        self.downsample_parameters = normalized_downsample_params
        self.use_legacy_padding = use_legacy_padding
        self.downsample_shapes: list[tuple[int, ...]] = []  # Track shapes before each downsample

        blocks: list[nn.Module] = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        output_channel = channels[0]
        downsample_idx = 0
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                # Record shape before downsampling (for decoder to restore exact size)
                blocks.append(_RecordShapeHook(self.downsample_shapes))
                # Use downsampling parameters for this level
                downsample_params = normalized_downsample_params[downsample_idx]
                blocks.append(
                    AEKLDownsample(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        kernel_size=downsample_params["kernel_size"],
                        stride=downsample_params["stride"],
                        padding=downsample_params["padding"],
                        use_legacy_padding=use_legacy_padding,
                    )
                )
                downsample_idx += 1

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )

            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=channels[-1],
                )
            )
        # Normalise and convert to latent size
        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=channels[-1], eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input through encoder blocks.

        Args:
            x: Input tensor.

        Returns:
            Encoded latent representation.
        """
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
        include_fc: whether to include the final linear layer. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        downsample_parameters: list of dicts with encoder downsampling parameters (strides).
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        downsample_parameters: list[dict] | dict | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        # Normalize downsampling parameters to get strides for upsampling
        num_downsample_levels = len(channels) - 1
        use_legacy_padding = downsample_parameters is None  # Track if using legacy defaults
        normalized_downsample_params = _normalize_downsample_parameters(
            downsample_parameters, num_downsample_levels, spatial_dims
        )

        # Will be populated by encoder with shapes before each downsample
        self.downsample_shapes: list[tuple[int, ...]] = []
        self.use_legacy_padding = use_legacy_padding

        reversed_block_out_channels = list(reversed(channels))

        blocks: list[nn.Module] = []

        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                AEKLResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]

        # Reverse downsample parameters for use during upsampling
        reversed_downsample_params = list(reversed(normalized_downsample_params))

        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    AEKLResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        SpatialAttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            include_fc=include_fc,
                            use_combined_linear=use_combined_linear,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                if use_convtranspose:
                    blocks.append(
                        Upsample(
                            spatial_dims=spatial_dims, mode="deconv", in_channels=block_in_ch, out_channels=block_in_ch
                        )
                    )
                else:
                    # For nontrainable upsampling: use exact target size from encoder
                    # This handles arbitrary input dimensions (odd, non-power-of-2, etc.)
                    post_conv = Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        out_channels=block_in_ch,
                        strides=1,
                        kernel_size=3,
                        padding=1,
                        conv_only=True,
                    )
                    # pass scale_factor from reversed_downsample_params as fallback
                    sf = tuple(reversed_downsample_params[i]["stride"])
                    blocks.append(
                        _ShapeRestoringUpsample(
                            spatial_dims=spatial_dims,
                            in_channels=block_in_ch,
                            out_channels=block_in_ch,
                            post_conv=post_conv,
                            shape_index=i,  # index into reversed downsample_shapes
                            downsample_shapes_ref=self.downsample_shapes,  # will be updated by AutoencoderKL
                            scale_factor=sf,
                        )
                    )

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward latent representation through decoder blocks.

        Args:
            x: Latent tensor.

        Returns:
            Reconstructed image tensor.
        """
        for block in self.blocks:
            x = block(x)
        return x


class AutoencoderKL(nn.Module):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see _ResBlock) per level.
        channels: number of output channels for each block.
        attention_levels: sequence of levels to add attention.
        latent_channels: latent embedding dimension.
        norm_num_groups: number of groups for the GroupNorm layers, channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_encoder_nonlocal_attn: if True use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: if True use non-local attention block in the decoder.
        use_checkpoint: if True, use activation checkpoint to save memory.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
        include_fc: whether to include the final linear layer in the attention block. Default to True.
        use_combined_linear: whether to use a single linear layer for qkv projection in the attention block, default to False.
        use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
            (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        downsample_parameters: downsampling parameters for each level. Can be:
            - None: use default (kernel_size=3, stride=2 for all levels)
            - dict: apply same parameters to all levels (e.g., {"kernel_size": (3,3,1), "stride": (2,2,1)})
            - list of dicts: one dict per downsampling level with keys "kernel_size", "stride", "padding"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_checkpoint: bool = False,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        downsample_parameters: list[dict] | dict | None = None,
    ) -> None:
        super().__init__()

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in channels):
            raise ValueError("AutoencoderKL expects all channels being multiple of norm_num_groups")

        if len(channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(channels))

        if len(num_res_blocks) != len(channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`channels`."
            )

        self.encoder: nn.Module = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            downsample_parameters=downsample_parameters,
        )

        # Get downsampling parameters from encoder to ensure decoder uses the same strides
        encoder_downsample_params = self.encoder.downsample_parameters

        self.decoder: nn.Module = Decoder(
            spatial_dims=spatial_dims,
            channels=channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            downsample_parameters=encoder_downsample_params,
        )

        # Link encoder shapes to decoder for exact size restoration
        # This must be done AFTER decoder creation so that _ShapeRestoringUpsample blocks
        # reference the shared list (not the empty list created during decoder init)
        self.decoder.downsample_shapes = self.encoder.downsample_shapes

        # Update all _ShapeRestoringUpsample blocks to reference the shared list
        for block in self.decoder.blocks:
            if isinstance(block, _ShapeRestoringUpsample):
                block.downsample_shapes_ref = self.encoder.downsample_shapes

        self.quant_conv_mu = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.quant_conv_log_sigma = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.post_quant_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.latent_channels = latent_channels
        self.use_checkpoint = use_checkpoint
        self._has_fresh_downsample_shapes = False

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        # Clear shape list before encoding to avoid unbounded growth across forward passes
        # Only clear if encoder supports shape tracking (e.g., Encoder class, not MaisiEncoder)
        if hasattr(self.encoder, "downsample_shapes"):
            cast(Encoder, self.encoder).downsample_shapes.clear()
            self._has_fresh_downsample_shapes = True

        if self.use_checkpoint:
            h = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            h = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        From the mean and sigma representations resulting of encoding an image through the latent space,
        obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
        adding the mean.

        Args:
            z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
            z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

        Returns:
            sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
        """
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes and decodes an input image.

        Args:
            x: BxCx[SPATIAL DIMENSIONS] tensor.

        Returns:
            reconstructed image, of the same shape as input
        """
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        # Clear stale encoder shapes if decode is called standalone (without preceding encode)
        # This ensures _ShapeRestoringUpsample blocks use scale_factor fallback for mismatched shapes
        if not self._has_fresh_downsample_shapes and hasattr(self.encoder, "downsample_shapes"):
            cast(Encoder, self.encoder).downsample_shapes.clear()

        z = self.post_quant_conv(z)
        dec: torch.Tensor
        try:
            if self.use_checkpoint:
                dec = torch.utils.checkpoint.checkpoint(self.decoder, z, use_reentrant=False)
            else:
                dec = self.decoder(z)
        finally:
            # Mark shapes as stale after decoding
            self._has_fresh_downsample_shapes = False
        return dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode, sample, and reconstruct an input image.

        Args:
            x: Input tensor of shape BxCx[SPATIAL_DIMS].

        Returns:
            Tuple containing:
                - reconstructed image
                - latent mean
                - latent standard deviation
        """
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)
        return reconstruction, z_mu, z_sigma

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image into latent space representation.

        Args:
            x: Input tensor.

        Returns:
            Sampled latent tensor.
        """
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation into image space.

        Args:
            z: Latent tensor.

        Returns:
            Decoded image tensor.
        """
        image = self.decode(z)
        return image

    def load_old_state_dict(self, old_state_dict: dict, verbose=False) -> None:
        """
        Load a state dict from an AutoencoderKL trained with [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels).

        Args:
            old_state_dict: state dict from the old AutoencoderKL model.
            verbose: if True, print diagnostic information about key mismatches.
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
        attention_blocks = [k.replace(".attn.to_q.weight", "") for k in new_state_dict if "attn.to_q.weight" in k]
        for block in attention_blocks:
            if f"{block}.to_q.weight" in old_state_dict:
                new_state_dict[f"{block}.attn.to_q.weight"] = old_state_dict.pop(f"{block}.to_q.weight")
            if f"{block}.to_k.weight" in old_state_dict:
                new_state_dict[f"{block}.attn.to_k.weight"] = old_state_dict.pop(f"{block}.to_k.weight")
            if f"{block}.to_v.weight" in old_state_dict:
                new_state_dict[f"{block}.attn.to_v.weight"] = old_state_dict.pop(f"{block}.to_v.weight")
            if f"{block}.to_q.bias" in old_state_dict:
                new_state_dict[f"{block}.attn.to_q.bias"] = old_state_dict.pop(f"{block}.to_q.bias")
            if f"{block}.to_k.bias" in old_state_dict:
                new_state_dict[f"{block}.attn.to_k.bias"] = old_state_dict.pop(f"{block}.to_k.bias")
            if f"{block}.to_v.bias" in old_state_dict:
                new_state_dict[f"{block}.attn.to_v.bias"] = old_state_dict.pop(f"{block}.to_v.bias")

            out_w = f"{block}.attn.out_proj.weight"
            out_b = f"{block}.attn.out_proj.bias"
            proj_w = f"{block}.proj_attn.weight"
            proj_b = f"{block}.proj_attn.bias"

            if out_w in new_state_dict:
                if proj_w in old_state_dict:
                    new_state_dict[out_w] = old_state_dict.pop(proj_w)
                    if proj_b in old_state_dict:
                        new_state_dict[out_b] = old_state_dict.pop(proj_b)
                    else:
                        new_state_dict[out_b] = torch.zeros(
                            new_state_dict[out_b].shape,
                            dtype=new_state_dict[out_b].dtype,
                            device=new_state_dict[out_b].device,
                        )
                else:
                    # No legacy proj_attn - initialize out_proj to identity/zero
                    new_state_dict[out_w] = torch.eye(
                        new_state_dict[out_w].shape[0],
                        dtype=new_state_dict[out_w].dtype,
                        device=new_state_dict[out_w].device,
                    )
                    new_state_dict[out_b] = torch.zeros(
                        new_state_dict[out_b].shape,
                        dtype=new_state_dict[out_b].dtype,
                        device=new_state_dict[out_b].device,
                    )
            elif proj_w in old_state_dict:
                # new model has no out_proj at all - discard the legacy keys so they
                # don't surface as "unexpected keys" during load_state_dict
                old_state_dict.pop(proj_w)
                old_state_dict.pop(proj_b, None)

        # fix the upsample conv blocks which were renamed postconv
        for k in new_state_dict:
            if "postconv" in k:
                old_name = k.replace("postconv", "conv")
                if old_name in old_state_dict:
                    new_state_dict[k] = old_state_dict.pop(old_name)
        if verbose:
            # print all remaining keys in old_state_dict
            print("remaining keys in old_state_dict:", old_state_dict.keys())
        self.load_state_dict(new_state_dict, strict=True)
