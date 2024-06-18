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

import gc
import monai
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence
from monai.networks.blocks import Convolution
from generative.networks.nets.autoencoderkl import (
    AttentionBlock,
    ResBlock,
    AutoencoderKL,
    Encoder,
)


NUM_SPLITS = 16

class InplaceGroupNorm3D(torch.nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(InplaceGroupNorm3D, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input):
        print("InplaceGroupNorm3D in", input.size())

        # Ensure the tensor is 5D: (N, C, D, H, W)
        if len(input.shape) != 5:
            raise ValueError("Expected a 5D tensor")

        N, C, D, H, W = input.shape

        # Reshape to (N, num_groups, C // num_groups, D, H, W)
        input = input.view(N, self.num_groups, C // self.num_groups, D, H, W)

        means, stds = [], []
        inputs = []
        for _i in range(input.size(1)):
            array = input[:, _i : _i + 1, ...]
            array = array.to(dtype=torch.float32)
            _mean = array.mean([2, 3, 4, 5], keepdim=True)
            _std = array.var([2, 3, 4, 5], unbiased=False, keepdim=True).add_(self.eps).sqrt_()

            _mean = _mean.to(dtype=torch.float32)
            _std = _std.to(dtype=torch.float32)

            inputs.append(array.sub_(_mean).div_(_std).to(dtype=torch.float16))

        del input
        torch.cuda.empty_cache()

        if max(inputs[0].size()) < 500:
            input = torch.cat([inputs[_k] for _k in range(len(inputs))], dim=1)
        else:
            _type = inputs[0].device.type
            if _type == "cuda":
                input = inputs[0].clone().to("cpu", non_blocking=True)
            else:
                input = inputs[0].clone()
            inputs[0] = 0
            torch.cuda.empty_cache()

            for _k in range(len(inputs) - 1):
                input = torch.cat((input, inputs[_k + 1].cpu()), dim=1)
                inputs[_k + 1] = 0
                torch.cuda.empty_cache()
                gc.collect()
                print(f"InplaceGroupNorm3D cat: {_k + 1}/{len(inputs) - 1}.")

            if _type == "cuda":
                input = input.to("cuda", non_blocking=True)

        # Reshape back to original size
        input = input.view(N, C, D, H, W)

        # Apply affine transformation if enabled
        if self.affine:
            input.mul_(self.weight.view(1, C, 1, 1, 1)).add_(self.bias.view(1, C, 1, 1, 1))

        print("InplaceGroupNorm3D out", input.size())

        return input


class StreamingConvolution(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super(StreamingConvolution, self).__init__()
        self.conv = monai.networks.blocks.convolutions.Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            adn_ordering,
            act,
            norm,
            dropout,
            dropout_dim,
            dilation,
            groups,
            bias,
            conv_only,
            is_transposed,
            padding,
            output_padding,
        )

        self.tp_dim = 1
        self.stride = strides[self.tp_dim] if isinstance(strides, list) else strides

    def forward(self, x):
        num_splits = NUM_SPLITS
        print("num_splits:", num_splits)
        l = x.size(self.tp_dim + 2)
        split_size = l // num_splits

        padding = 3
        if padding % self.stride > 0:
            padding = (padding // self.stride + 1) * self.stride
        print("padding:", padding)

        overlaps = [0] + [padding] * (num_splits - 1)
        last_padding = x.size(self.tp_dim + 2) % split_size

        if self.tp_dim == 0:
            splits = [
                x[
                    :,
                    :,
                    i * split_size
                    - overlaps[i] : (i + 1) * split_size
                    + (padding if i != num_splits - 1 else last_padding),
                    :,
                    :,
                ]
                for i in range(num_splits)
            ]
        elif self.tp_dim == 1:
            splits = [
                x[
                    :,
                    :,
                    :,
                    i * split_size
                    - overlaps[i] : (i + 1) * split_size
                    + (padding if i != num_splits - 1 else last_padding),
                    :,
                ]
                for i in range(num_splits)
            ]
        elif self.tp_dim == 2:
            splits = [
                x[
                    :,
                    :,
                    :,
                    :,
                    i * split_size
                    - overlaps[i] : (i + 1) * split_size
                    + (padding if i != num_splits - 1 else last_padding),
                ]
                for i in range(num_splits)
            ]

        for _j in range(len(splits)):
            print(f"splits {_j + 1}/{len(splits)}:", splits[_j].size())

        del x
        torch.cuda.empty_cache()

        splits_0_size = list(splits[0].size())
        print("splits_0_size:", splits_0_size)

        outputs = []
        _type = splits[0].device.type
        for _i in range(num_splits):
            outputs.append(self.conv(splits[_i]))

            splits[_i] = 0
            torch.cuda.empty_cache()

        for _j in range(len(outputs)):
            print(f"outputs before {_j + 1}/{len(outputs)}:", outputs[_j].size())

        del splits
        torch.cuda.empty_cache()

        split_size_out = split_size
        padding_s = padding
        non_tp_dim = self.tp_dim + 1 if self.tp_dim < 2 else 0
        if outputs[0].size(non_tp_dim + 2) // splits_0_size[non_tp_dim + 2] == 2:
            split_size_out *= 2
            padding_s *= 2
        elif splits_0_size[non_tp_dim + 2] // outputs[0].size(non_tp_dim + 2) == 2:
            split_size_out = split_size_out // 2
            padding_s = padding_s // 2

        if self.tp_dim == 0:
            outputs[0] = outputs[0][:, :, :split_size_out, :, :]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, padding_s : padding_s + split_size_out, :, :]
        elif self.tp_dim == 1:
            print("outputs", outputs[0].size(3), f"padding_s: 0, {split_size_out}")
            outputs[0] = outputs[0][:, :, :, :split_size_out, :]
            for i in range(1, num_splits):
                print(
                    "outputs",
                    outputs[i].size(3),
                    f"padding_s: {padding_s}, {padding_s + split_size_out}",
                )
                outputs[i] = outputs[i][:, :, :, padding_s : padding_s + split_size_out, :]
        elif self.tp_dim == 2:
            outputs[0] = outputs[0][:, :, :, :, :split_size_out]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, :, :, padding_s : padding_s + split_size_out]

        for i in range(num_splits):
            print(f"outputs after {i + 1}/{len(outputs)}:", outputs[i].size())

        if max(outputs[0].size()) < 500:
            print(f"outputs[0].device.type: {outputs[0].device.type}.")
            x = torch.cat([out for out in outputs], dim=self.tp_dim + 2)
        else:
            _type = outputs[0].device.type
            if _type == "cuda":
                x = outputs[0].clone().to("cpu", non_blocking=True)
            outputs[0] = 0
            torch.cuda.empty_cache()
            for _k in range(len(outputs) - 1):
                x = torch.cat((x, outputs[_k + 1].cpu()), dim=self.tp_dim + 2)
                outputs[_k + 1] = 0
                torch.cuda.empty_cache()
                gc.collect()
                print(f"StreamingConvolution cat: {_k + 1}/{len(outputs) - 1}.")
            if _type == "cuda":
                x = x.to("cuda", non_blocking=True)

        del outputs
        torch.cuda.empty_cache()

        return x


class StreamingUpsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
    """

    def __init__(self, spatial_dims: int, in_channels: int, use_convtranspose: bool) -> None:
        super().__init__()
        if use_convtranspose:
            self.conv = StreamingConvolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=2,
                kernel_size=3,
                padding=1,
                conv_only=True,
                is_transposed=True,
            )
        else:
            self.conv = StreamingConvolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        self.use_convtranspose = use_convtranspose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtranspose:
            return self.conv(x)

        dtype = x.dtype

        x = F.interpolate(x, scale_factor=2.0, mode="trilinear")
        torch.cuda.empty_cache()

        x = self.conv(x)
        torch.cuda.empty_cache()

        return x


class StreamingDownsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
    """

    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims

        self.conv = StreamingConvolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class StreamingResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int,
        norm_eps: float,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = InplaceGroupNorm3D(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )
        self.conv1 = StreamingConvolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = InplaceGroupNorm3D(
            num_groups=norm_num_groups,
            num_channels=out_channels,
            eps=norm_eps,
            affine=True,
        )
        self.conv2 = StreamingConvolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = StreamingConvolution(
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
        torch.cuda.empty_cache()

        h = F.silu(h)
        torch.cuda.empty_cache()
        h = self.conv1(h)
        torch.cuda.empty_cache()

        h = self.norm2(h)
        torch.cuda.empty_cache()

        h = F.silu(h)
        torch.cuda.empty_cache()
        h = self.conv2(h)
        torch.cuda.empty_cache()

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
            torch.cuda.empty_cache()

        return x + h


class StreamingEncoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        num_channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
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
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        blocks = []

        # Initial convolution
        blocks.append(
            StreamingConvolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    StreamingResBlock(
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
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(StreamingDownsample(spatial_dims=spatial_dims, in_channels=input_channel))

        # Non-local attention block
        if with_nonlocal_attn is True:
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
        # Normalise and convert to latent size
        blocks.append(
            InplaceGroupNorm3D(
                num_groups=norm_num_groups,
                num_channels=num_channels[-1],
                eps=norm_eps,
                affine=True,
            )
        )
        blocks.append(
            StreamingConvolution(
                spatial_dims=self.spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            torch.cuda.empty_cache()
        return x


class StreamingDecoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
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
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_convtranspose: bool = False,
        tp_dim: int = 1,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        self.tp_dim = tp_dim

        reversed_block_out_channels = list(reversed(num_channels))

        blocks = []

        # Initial convolution
        blocks.append(
            StreamingConvolution(
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
                    StreamingResBlock(
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
                    StreamingUpsample(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        use_convtranspose=use_convtranspose,
                    )
                )

        blocks.append(
            InplaceGroupNorm3D(
                num_groups=norm_num_groups,
                num_channels=block_in_ch,
                eps=norm_eps,
                affine=True,
            )
        )
        blocks.append(
            StreamingConvolution(
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
        for _i in range(len(self.blocks)):
            block = self.blocks[_i]
            print(block, type(block), type(type(block)))

            if _i < len(self.blocks) - 0:
                x = block(x)
                torch.cuda.empty_cache()
            else:
                num_splits = NUM_SPLITS
                print("num_splits:", num_splits)

                l = x.size(self.tp_dim + 2)
                split_size = l // num_splits

                padding = 3
                print("padding:", padding)

                overlaps = [0] + [padding] * (num_splits - 1)
                if self.tp_dim == 0:
                    splits = [
                        x[
                            :,
                            :,
                            i * split_size
                            - overlaps[i] : (i + 1) * split_size
                            + (padding if i != num_splits - 1 else 0),
                            :,
                            :,
                        ]
                        for i in range(num_splits)
                    ]
                elif self.tp_dim == 1:
                    splits = [
                        x[
                            :,
                            :,
                            :,
                            i * split_size
                            - overlaps[i] : (i + 1) * split_size
                            + (padding if i != num_splits - 1 else 0),
                            :,
                        ]
                        for i in range(num_splits)
                    ]
                elif self.tp_dim == 2:
                    splits = [
                        x[
                            :,
                            :,
                            :,
                            :,
                            i * split_size
                            - overlaps[i] : (i + 1) * split_size
                            + (padding if i != num_splits - 1 else 0),
                        ]
                        for i in range(num_splits)
                    ]

                for _j in range(len(splits)):
                    print(f"splits {_j + 1}/{len(splits)}:", splits[_j].size())

                del x
                torch.cuda.empty_cache()

                outputs = [block(splits[i]) for i in range(num_splits)]

                del splits
                torch.cuda.empty_cache()

                split_size_out = split_size
                padding_s = padding
                non_tp_dim = self.tp_dim + 1 if self.tp_dim < 2 else 0
                if outputs[0].size(non_tp_dim + 2) // splits[0].size(non_tp_dim + 2) == 2:
                    split_size_out *= 2
                    padding_s *= 2
                print("split_size_out:", split_size_out)
                print("padding_s:", padding_s)

                if self.tp_dim == 0:
                    outputs[0] = outputs[0][:, :, :split_size_out, :, :]
                    for i in range(1, num_splits):
                        outputs[i] = outputs[i][:, :, padding_s : padding_s + split_size_out, :, :]
                elif self.tp_dim == 1:
                    print("outputs", outputs[0].size(3), f"padding_s: 0, {split_size_out}")
                    outputs[0] = outputs[0][:, :, :, :split_size_out, :]
                    for i in range(1, num_splits):
                        print(
                            "outputs",
                            outputs[i].size(3),
                            f"padding_s: {padding_s}, {padding_s + split_size_out}",
                        )
                        outputs[i] = outputs[i][:, :, :, padding_s : padding_s + split_size_out, :]
                elif self.tp_dim == 2:
                    outputs[0] = outputs[0][:, :, :, :, :split_size_out]
                    for i in range(1, num_splits):
                        outputs[i] = outputs[i][:, :, :, :, padding_s : padding_s + split_size_out]

                for i in range(num_splits):
                    print(f"outputs after {i + 1}/{len(outputs)}:", outputs[i].size())

                if max(outputs[0].size()) < 500:
                    x = torch.cat([out for out in outputs], dim=self.tp_dim + 2)
                else:
                    x = outputs[0].clone().to("cpu", non_blocking=True)
                    outputs[0] = 0
                    torch.cuda.empty_cache()
                    for _k in range(len(outputs) - 1):
                        x = torch.cat((x, outputs[_k + 1].cpu()), dim=self.tp_dim + 2)
                        outputs[_k + 1] = 0
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"cat: {_k + 1}/{len(outputs) - 1}.")
                    x = x.to("cuda", non_blocking=True)

                del outputs
                torch.cuda.empty_cache()

        return x


class StreamingAutoencoderKL(AutoencoderKL):
    """
    Override encoder to make it align with original ldm codebase and support activation checkpointing.
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
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
        use_convtranspose: bool = False,
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

        self.encoder = StreamingEncoder(
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
        )

        # Override decoder using transposed conv
        self.decoder = StreamingDecoder(
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
        )
