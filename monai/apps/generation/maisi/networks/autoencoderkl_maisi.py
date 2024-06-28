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

import gc
import logging
from typing import TYPE_CHECKING, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution
from monai.utils import optional_import
from monai.utils.type_conversion import convert_to_tensor

AttentionBlock, has_attentionblock = optional_import("generative.networks.nets.autoencoderkl", name="AttentionBlock")
AutoencoderKL, has_autoencoderkl = optional_import("generative.networks.nets.autoencoderkl", name="AutoencoderKL")
ResBlock, has_resblock = optional_import("generative.networks.nets.autoencoderkl", name="ResBlock")


if TYPE_CHECKING:
    from generative.networks.nets.autoencoderkl import AutoencoderKL as AutoencoderKLType
else:
    AutoencoderKLType = cast(type, AutoencoderKL)


# Set up logging configuration
logger = logging.getLogger(__name__)


def _empty_cuda_cache(save_mem: bool) -> None:
    if torch.cuda.is_available() and save_mem:
        torch.cuda.empty_cache()
    return


class MaisiGroupNorm3D(nn.GroupNorm):
    """
    Custom 3D Group Normalization with optional print_info output.

    Args:
        num_groups: Number of groups for the group norm.
        num_channels: Number of channels for the group norm.
        eps: Epsilon value for numerical stability.
        affine: Whether to use learnable affine parameters, default to `True`.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        norm_float16: bool = False,
        print_info: bool = False,
        save_mem: bool = True,
    ):
        super().__init__(num_groups, num_channels, eps, affine)
        self.norm_float16 = norm_float16
        self.print_info = print_info
        self.save_mem = save_mem

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.print_info:
            logger.info(f"MaisiGroupNorm3D with input size: {input.size()}")

        if len(input.shape) != 5:
            raise ValueError("Expected a 5D tensor")

        param_n, param_c, param_d, param_h, param_w = input.shape
        input = input.view(param_n, self.num_groups, param_c // self.num_groups, param_d, param_h, param_w)

        inputs = []
        for i in range(input.size(1)):
            array = input[:, i : i + 1, ...].to(dtype=torch.float32)
            mean = array.mean([2, 3, 4, 5], keepdim=True)
            std = array.var([2, 3, 4, 5], unbiased=False, keepdim=True).add_(self.eps).sqrt_()
            if self.norm_float16:
                inputs.append(((array - mean) / std).to(dtype=torch.float16))
            else:
                inputs.append((array - mean) / std)

        del input
        _empty_cuda_cache(self.save_mem)

        input = torch.cat(inputs, dim=1) if max(inputs[0].size()) < 500 else self._cat_inputs(inputs)

        input = input.view(param_n, param_c, param_d, param_h, param_w)
        if self.affine:
            input.mul_(self.weight.view(1, param_c, 1, 1, 1)).add_(self.bias.view(1, param_c, 1, 1, 1))

        if self.print_info:
            logger.info(f"MaisiGroupNorm3D with output size: {input.size()}")

        return input

    def _cat_inputs(self, inputs):
        input_type = inputs[0].device.type
        input = inputs[0].clone().to("cpu", non_blocking=True) if input_type == "cuda" else inputs[0].clone()
        inputs[0] = 0
        _empty_cuda_cache(self.save_mem)

        for k in range(len(inputs) - 1):
            input = torch.cat((input, inputs[k + 1].cpu()), dim=1)
            inputs[k + 1] = 0
            _empty_cuda_cache(self.save_mem)
            gc.collect()

            if self.print_info:
                logger.info(f"MaisiGroupNorm3D concat progress: {k + 1}/{len(inputs) - 1}.")

        return input.to("cuda", non_blocking=True) if input_type == "cuda" else input


class MaisiConvolution(nn.Module):
    """
    Convolutional layer with optional print_info output and custom splitting mechanism.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        print_info: Whether to print information.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
        Additional arguments for the convolution operation.
        https://docs.monai.io/en/stable/networks.html#convolution
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_splits: int,
        dim_split: int,
        print_info: bool,
        save_mem: bool = True,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=kernel_size,
            adn_ordering=adn_ordering,
            act=act,
            norm=norm,
            dropout=dropout,
            dropout_dim=dropout_dim,
            dilation=dilation,
            groups=groups,
            bias=bias,
            conv_only=conv_only,
            is_transposed=is_transposed,
            padding=padding,
            output_padding=output_padding,
        )

        self.dim_split = dim_split
        self.stride = strides[self.dim_split] if isinstance(strides, list) else strides
        self.num_splits = num_splits
        self.print_info = print_info
        self.save_mem = save_mem

    def _split_tensor(self, x: torch.Tensor, split_size: int, padding: int) -> list[torch.Tensor]:
        overlaps = [0] + [padding] * (self.num_splits - 1)
        last_padding = x.size(self.dim_split + 2) % split_size

        slices = [slice(None)] * 5
        splits: list[torch.Tensor] = []
        for i in range(self.num_splits):
            slices[self.dim_split + 2] = slice(
                i * split_size - overlaps[i],
                (i + 1) * split_size + (padding if i != self.num_splits - 1 else last_padding),
            )
            splits.append(x[tuple(slices)])

        if self.print_info:
            for j in range(len(splits)):
                logger.info(f"Split {j + 1}/{len(splits)} size: {splits[j].size()}")

        return splits

    def _concatenate_tensors(self, outputs: list[torch.Tensor], split_size: int, padding: int) -> torch.Tensor:
        slices = [slice(None)] * 5
        for i in range(self.num_splits):
            slices[self.dim_split + 2] = slice(None, split_size) if i == 0 else slice(padding, padding + split_size)
            outputs[i] = outputs[i][tuple(slices)]

        if self.print_info:
            for i in range(self.num_splits):
                logger.info(f"Output {i + 1}/{len(outputs)} size after: {outputs[i].size()}")

        if max(outputs[0].size()) < 500:
            x = torch.cat(outputs, dim=self.dim_split + 2)
        else:
            x = outputs[0].clone().to("cpu", non_blocking=True)
            outputs[0] = torch.Tensor(0)
            _empty_cuda_cache(self.save_mem)
            for k in range(len(outputs) - 1):
                x = torch.cat((x, outputs[k + 1].cpu()), dim=self.dim_split + 2)
                outputs[k + 1] = torch.Tensor(0)
                _empty_cuda_cache(self.save_mem)
                gc.collect()
                if self.print_info:
                    logger.info(f"MaisiConvolution concat progress: {k + 1}/{len(outputs) - 1}.")

            x = x.to("cuda", non_blocking=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.print_info:
            logger.info(f"Number of splits: {self.num_splits}")

        # compute size of splits
        l = x.size(self.dim_split + 2)
        split_size = l // self.num_splits

        # update padding length if necessary
        padding = 3
        if padding % self.stride > 0:
            padding = (padding // self.stride + 1) * self.stride
        if self.print_info:
            logger.info(f"Padding size: {padding}")

        # split tensor into a list of tensors
        splits = self._split_tensor(x, split_size, padding)

        del x
        _empty_cuda_cache(self.save_mem)

        # convolution
        outputs = [self.conv(split) for split in splits]
        if self.print_info:
            for j in range(len(outputs)):
                logger.info(f"Output {j + 1}/{len(outputs)} size before: {outputs[j].size()}")

        # update size of splits and padding length for output
        split_size_out = split_size
        padding_s = padding
        non_dim_split = self.dim_split + 1 if self.dim_split < 2 else 0
        if outputs[0].size(non_dim_split + 2) // splits[0].size(non_dim_split + 2) == 2:
            split_size_out *= 2
            padding_s *= 2
        elif splits[0].size(non_dim_split + 2) // outputs[0].size(non_dim_split + 2) == 2:
            split_size_out //= 2
            padding_s //= 2

        # concatenate list of tensors
        x = self._concatenate_tensors(outputs, split_size_out, padding_s)

        del outputs
        _empty_cuda_cache(self.save_mem)

        return x


class MaisiUpsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels to the layer.
        use_convtranspose: If True, use ConvTranspose to upsample feature maps in decoder.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        print_info: Whether to print information.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        use_convtranspose: bool,
        num_splits: int,
        dim_split: int,
        print_info: bool,
        save_mem: bool = True,
    ) -> None:
        super().__init__()
        self.conv = MaisiConvolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2 if use_convtranspose else 1,
            kernel_size=3,
            padding=1,
            conv_only=True,
            is_transposed=use_convtranspose,
            num_splits=num_splits,
            dim_split=dim_split,
            print_info=print_info,
            save_mem=save_mem,
        )
        self.use_convtranspose = use_convtranspose
        self.save_mem = save_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtranspose:
            x = self.conv(x)
            x_tensor: torch.Tensor = convert_to_tensor(x)
            return x_tensor

        x = F.interpolate(x, scale_factor=2.0, mode="trilinear")
        _empty_cuda_cache(self.save_mem)
        x = self.conv(x)
        _empty_cuda_cache(self.save_mem)

        out_tensor: torch.Tensor = convert_to_tensor(x)
        return out_tensor


class MaisiDownsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        print_info: Whether to print information.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_splits: int,
        dim_split: int,
        print_info: bool,
        save_mem: bool = True,
    ) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims
        self.conv = MaisiConvolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
            num_splits=num_splits,
            dim_split=dim_split,
            print_info=print_info,
            save_mem=save_mem,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class MaisiResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Input channels to the layer.
        norm_num_groups: Number of groups for the group norm layer.
        norm_eps: Epsilon for the normalization.
        out_channels: Number of output channels.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int,
        norm_eps: float,
        out_channels: int,
        num_splits: int,
        dim_split: int,
        norm_float16: bool = False,
        print_info: bool = False,
        save_mem: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.save_mem = save_mem

        self.norm1 = MaisiGroupNorm3D(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )
        self.conv1 = MaisiConvolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
            num_splits=num_splits,
            dim_split=dim_split,
            print_info=print_info,
            save_mem=save_mem,
        )
        self.norm2 = MaisiGroupNorm3D(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )
        self.conv2 = MaisiConvolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
            num_splits=num_splits,
            dim_split=dim_split,
            print_info=print_info,
            save_mem=save_mem,
        )

        self.nin_shortcut = (
            MaisiConvolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
                num_splits=num_splits,
                dim_split=dim_split,
                print_info=print_info,
                save_mem=save_mem,
            )
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        _empty_cuda_cache(self.save_mem)

        h = F.silu(h)
        _empty_cuda_cache(self.save_mem)
        h = self.conv1(h)
        _empty_cuda_cache(self.save_mem)

        h = self.norm2(h)
        _empty_cuda_cache(self.save_mem)

        h = F.silu(h)
        _empty_cuda_cache(self.save_mem)
        h = self.conv2(h)
        _empty_cuda_cache(self.save_mem)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
            _empty_cuda_cache(self.save_mem)

        out = x + h
        out_tensor: torch.Tensor = convert_to_tensor(out)
        return out_tensor


class MaisiEncoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        num_channels: Sequence of block output channels.
        out_channels: Number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: Number of residual blocks (see ResBlock) per level.
        norm_num_groups: Number of groups for the group norm layers.
        norm_eps: Epsilon for the normalization.
        attention_levels: Indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: If True, use non-local attention block.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        num_splits: int,
        dim_split: int,
        norm_float16: bool = False,
        print_info: bool = False,
        save_mem: bool = True,
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        # Check if attention_levels and num_channels have the same size
        if len(attention_levels) != len(num_channels):
            raise ValueError("attention_levels and num_channels must have the same size")

        # Check if num_res_blocks and num_channels have the same size
        if len(num_res_blocks) != len(num_channels):
            raise ValueError("num_res_blocks and num_channels must have the same size")

        self.save_mem = save_mem

        blocks: list[nn.Module] = []

        blocks.append(
            MaisiConvolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                num_splits=num_splits,
                dim_split=dim_split,
                print_info=print_info,
                save_mem=save_mem,
            )
        )

        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(num_res_blocks[i]):
                blocks.append(
                    MaisiResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                        num_splits=num_splits,
                        dim_split=dim_split,
                        norm_float16=norm_float16,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(
                    MaisiDownsample(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        num_splits=num_splits,
                        dim_split=dim_split,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )

        if with_nonlocal_attn:
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                )
            )

            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                )
            )

        blocks.append(
            MaisiGroupNorm3D(
                num_groups=norm_num_groups,
                num_channels=num_channels[-1],
                eps=norm_eps,
                affine=True,
                norm_float16=norm_float16,
                print_info=print_info,
                save_mem=save_mem,
            )
        )
        blocks.append(
            MaisiConvolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                num_splits=num_splits,
                dim_split=dim_split,
                print_info=print_info,
                save_mem=save_mem,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            _empty_cuda_cache(self.save_mem)
        return x


class MaisiDecoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        num_channels: Sequence of block output channels.
        in_channels: Number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks (see ResBlock) per level.
        norm_num_groups: Number of groups for the group norm layers.
        norm_eps: Epsilon for the normalization.
        attention_levels: Indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: If True, use non-local attention block.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        use_convtranspose: If True, use ConvTranspose to upsample feature maps in decoder.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        num_splits: int,
        dim_split: int,
        norm_float16: bool = False,
        print_info: bool = False,
        save_mem: bool = True,
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_convtranspose: bool = False,
    ) -> None:
        super().__init__()
        self.print_info = print_info
        self.save_mem = save_mem

        reversed_block_out_channels = list(reversed(num_channels))

        blocks: list[nn.Module] = []

        blocks.append(
            MaisiConvolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                num_splits=num_splits,
                dim_split=dim_split,
                print_info=print_info,
                save_mem=save_mem,
            )
        )

        if with_nonlocal_attn:
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                ResBlock(
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
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    MaisiResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                        num_splits=num_splits,
                        dim_split=dim_split,
                        norm_float16=norm_float16,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(
                    MaisiUpsample(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        use_convtranspose=use_convtranspose,
                        num_splits=num_splits,
                        dim_split=dim_split,
                        print_info=print_info,
                        save_mem=save_mem,
                    )
                )

        blocks.append(
            MaisiGroupNorm3D(
                num_groups=norm_num_groups,
                num_channels=block_in_ch,
                eps=norm_eps,
                affine=True,
                norm_float16=norm_float16,
                print_info=print_info,
                save_mem=save_mem,
            )
        )
        blocks.append(
            MaisiConvolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
                num_splits=num_splits,
                dim_split=dim_split,
                print_info=print_info,
                save_mem=save_mem,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            _empty_cuda_cache(self.save_mem)
        return x


class AutoencoderKlMaisi(AutoencoderKLType):
    """
    AutoencoderKL with custom MaisiEncoder and MaisiDecoder.

    Args:
        spatial_dims: Number of spatial dimensions (1D, 2D, 3D).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks per level.
        num_channels: Sequence of block output channels.
        attention_levels: Indicate which level from num_channels contain an attention block.
        latent_channels: Number of channels in the latent space.
        norm_num_groups: Number of groups for the group norm layers.
        norm_eps: Epsilon for the normalization.
        with_encoder_nonlocal_attn: If True, use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: If True, use non-local attention block in the decoder.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        use_checkpointing: If True, use activation checkpointing.
        use_convtranspose: If True, use ConvTranspose to upsample feature maps in decoder.
        num_splits: Number of splits for the input tensor.
        dim_split: Dimension of splitting for the input tensor.
        norm_float16: If True, convert output of MaisiGroupNorm3D to float16 format, default to `False`.
        print_info: Whether to print information, default to `False`.
        save_mem: Whether to clean CUDA cache in order to save GPU memory, default to `True`.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        num_channels: Sequence[int],
        attention_levels: Sequence[bool],
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = False,
        with_decoder_nonlocal_attn: bool = False,
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
        use_convtranspose: bool = False,
        num_splits: int = 16,
        dim_split: int = 0,
        norm_float16: bool = False,
        print_info: bool = False,
        save_mem: bool = True,
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            num_res_blocks,
            num_channels,
            attention_levels,
            latent_channels,
            norm_num_groups,
            norm_eps,
            with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn,
            use_flash_attention,
            use_checkpointing,
            use_convtranspose,
        )

        self.encoder = MaisiEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )

        self.decoder = MaisiDecoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            print_info=print_info,
            save_mem=save_mem,
        )
