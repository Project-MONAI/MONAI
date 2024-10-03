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
# https://github.com/MIC-DKFZ/MedNeXt
# and are used under the terms of the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.mednext_block import MedNeXtBlock, MedNeXtDownBlock, MedNeXtOutBlock, MedNeXtUpBlock

__all__ = [
    "MedNeXt",
    "MedNeXtSmall",
    "MedNeXtBase",
    "MedNeXtMedium",
    "MedNeXtLarge",
    "MedNext",
    "MedNextS",
    "MedNeXtS",
    "MedNextSmall",
    "MedNextB",
    "MedNeXtB",
    "MedNextBase",
    "MedNextM",
    "MedNeXtM",
    "MedNextMedium",
    "MedNextL",
    "MedNeXtL",
    "MedNextLarge",
]


class MedNeXt(nn.Module):
    """
    MedNeXt model class from paper: https://arxiv.org/pdf/2303.09975

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        encoder_expansion_ratio: expansion ratio for encoder blocks. Defaults to 2.
        decoder_expansion_ratio: expansion ratio for decoder blocks. Defaults to 2.
        bottleneck_expansion_ratio: expansion ratio for bottleneck blocks. Defaults to 2.
        kernel_size: kernel size for convolutions. Defaults to 7.
        deep_supervision: whether to use deep supervision. Defaults to False.
        use_residual_connection: whether to use residual connections in standard, down and up blocks. Defaults to False.
        blocks_down: number of blocks in each encoder stage. Defaults to [2, 2, 2, 2].
        blocks_bottleneck: number of blocks in bottleneck stage. Defaults to 2.
        blocks_up: number of blocks in each decoder stage. Defaults to [2, 2, 2, 2].
        norm_type: type of normalization layer. Defaults to 'group'.
        global_resp_norm: whether to use Global Response Normalization. Defaults to False. Refer: https://arxiv.org/abs/2301.00808
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 2,
        encoder_expansion_ratio: Sequence[int] | int = 2,
        decoder_expansion_ratio: Sequence[int] | int = 2,
        bottleneck_expansion_ratio: int = 2,
        kernel_size: int = 7,
        deep_supervision: bool = False,
        use_residual_connection: bool = False,
        blocks_down: Sequence[int] = (2, 2, 2, 2),
        blocks_bottleneck: int = 2,
        blocks_up: Sequence[int] = (2, 2, 2, 2),
        norm_type: str = "group",
        global_resp_norm: bool = False,
    ):
        """
        Initialize the MedNeXt model.

        This method sets up the architecture of the model, including:
        - Stem convolution
        - Encoder stages and downsampling blocks
        - Bottleneck blocks
        - Decoder stages and upsampling blocks
        - Output blocks for deep supervision (if enabled)
        """
        super().__init__()

        self.do_ds = deep_supervision
        assert spatial_dims in [2, 3], "`spatial_dims` can only be 2 or 3."
        spatial_dims_str = f"{spatial_dims}d"
        enc_kernel_size = dec_kernel_size = kernel_size

        if isinstance(encoder_expansion_ratio, int):
            encoder_expansion_ratio = [encoder_expansion_ratio] * len(blocks_down)

        if isinstance(decoder_expansion_ratio, int):
            decoder_expansion_ratio = [decoder_expansion_ratio] * len(blocks_up)

        conv = nn.Conv2d if spatial_dims_str == "2d" else nn.Conv3d

        self.stem = conv(in_channels, init_filters, kernel_size=1)

        enc_stages = []
        down_blocks = []

        for i, num_blocks in enumerate(blocks_down):
            enc_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=init_filters * (2**i),
                            out_channels=init_filters * (2**i),
                            expansion_ratio=encoder_expansion_ratio[i],
                            kernel_size=enc_kernel_size,
                            use_residual_connection=use_residual_connection,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            global_resp_norm=global_resp_norm,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

            down_blocks.append(
                MedNeXtDownBlock(
                    in_channels=init_filters * (2**i),
                    out_channels=init_filters * (2 ** (i + 1)),
                    expansion_ratio=encoder_expansion_ratio[i],
                    kernel_size=enc_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                )
            )

        self.enc_stages = nn.ModuleList(enc_stages)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=init_filters * (2 ** len(blocks_down)),
                    out_channels=init_filters * (2 ** len(blocks_down)),
                    expansion_ratio=bottleneck_expansion_ratio,
                    kernel_size=dec_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    global_resp_norm=global_resp_norm,
                )
                for _ in range(blocks_bottleneck)
            ]
        )

        up_blocks = []
        dec_stages = []
        for i, num_blocks in enumerate(blocks_up):
            up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=init_filters * (2 ** (len(blocks_up) - i)),
                    out_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                    expansion_ratio=decoder_expansion_ratio[i],
                    kernel_size=dec_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    global_resp_norm=global_resp_norm,
                )
            )

            dec_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            out_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            expansion_ratio=decoder_expansion_ratio[i],
                            kernel_size=dec_kernel_size,
                            use_residual_connection=use_residual_connection,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            global_resp_norm=global_resp_norm,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.out_0 = MedNeXtOutBlock(in_channels=init_filters, n_classes=out_channels, dim=spatial_dims_str)

        if deep_supervision:
            out_blocks = [
                MedNeXtOutBlock(in_channels=init_filters * (2**i), n_classes=out_channels, dim=spatial_dims_str)
                for i in range(1, len(blocks_up) + 1)
            ]

            out_blocks.reverse()
            self.out_blocks = nn.ModuleList(out_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Sequence[torch.Tensor]:
        """
        Forward pass of the MedNeXt model.

        This method performs the forward pass through the model, including:
        - Stem convolution
        - Encoder stages and downsampling
        - Bottleneck blocks
        - Decoder stages and upsampling with skip connections
        - Output blocks for deep supervision (if enabled)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor or Sequence[torch.Tensor]: Output tensor(s).
        """
        # Apply stem convolution
        x = self.stem(x)

        # Encoder forward pass
        enc_outputs = []
        for enc_stage, down_block in zip(self.enc_stages, self.down_blocks):
            x = enc_stage(x)
            enc_outputs.append(x)
            x = down_block(x)

        # Bottleneck forward pass
        x = self.bottleneck(x)

        # Initialize deep supervision outputs if enabled
        if self.do_ds:
            ds_outputs = []

        # Decoder forward pass with skip connections
        for i, (up_block, dec_stage) in enumerate(zip(self.up_blocks, self.dec_stages)):
            if self.do_ds and i < len(self.out_blocks):
                ds_outputs.append(self.out_blocks[i](x))

            x = up_block(x)
            x = x + enc_outputs[-(i + 1)]
            x = dec_stage(x)

        # Final output block
        x = self.out_0(x)

        # Return output(s)
        if self.do_ds and self.training:
            return (x, *ds_outputs[::-1])
        else:
            return x


# Define the MedNeXt variants as reported in 10.48550/arXiv.2303.09975
def create_mednext(
    variant: str,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    kernel_size: int = 3,
    deep_supervision: bool = False,
) -> MedNeXt:
    """
    Factory method to create MedNeXt variants.

    Args:
        variant (str): The MedNeXt variant to create ('S', 'B', 'M', or 'L').
        spatial_dims (int): Number of spatial dimensions. Defaults to 3.
        in_channels (int): Number of input channels. Defaults to 1.
        out_channels (int): Number of output channels. Defaults to 2.
        kernel_size (int): Kernel size for convolutions. Defaults to 3.
        deep_supervision (bool): Whether to use deep supervision. Defaults to False.

    Returns:
        MedNeXt: The specified MedNeXt variant.

    Raises:
        ValueError: If an invalid variant is specified.
    """
    common_args = {
        "spatial_dims": spatial_dims,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "deep_supervision": deep_supervision,
        "use_residual_connection": True,
        "norm_type": "group",
        "global_resp_norm": False,
        "init_filters": 32,
    }

    if variant.upper() == "S":
        return MedNeXt(
            encoder_expansion_ratio=2,
            decoder_expansion_ratio=2,
            bottleneck_expansion_ratio=2,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            **common_args,  # type: ignore
        )
    elif variant.upper() == "B":
        return MedNeXt(
            encoder_expansion_ratio=(2, 3, 4, 4),
            decoder_expansion_ratio=(4, 4, 3, 2),
            bottleneck_expansion_ratio=4,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            **common_args,  # type: ignore
        )
    elif variant.upper() == "M":
        return MedNeXt(
            encoder_expansion_ratio=(2, 3, 4, 4),
            decoder_expansion_ratio=(4, 4, 3, 2),
            bottleneck_expansion_ratio=4,
            blocks_down=(3, 4, 4, 4),
            blocks_bottleneck=4,
            blocks_up=(4, 4, 4, 3),
            **common_args,  # type: ignore
        )
    elif variant.upper() == "L":
        return MedNeXt(
            encoder_expansion_ratio=(3, 4, 8, 8),
            decoder_expansion_ratio=(8, 8, 4, 3),
            bottleneck_expansion_ratio=8,
            blocks_down=(3, 4, 8, 8),
            blocks_bottleneck=8,
            blocks_up=(8, 8, 4, 3),
            **common_args,  # type: ignore
        )
    else:
        raise ValueError(f"Invalid MedNeXt variant: {variant}")


MedNext = MedNeXt
MedNextS = MedNeXtS = MedNextSmall = MedNeXtSmall = lambda **kwargs: create_mednext("S", **kwargs)
MedNextB = MedNeXtB = MedNextBase = MedNeXtBase = lambda **kwargs: create_mednext("B", **kwargs)
MedNextM = MedNeXtM = MedNextMedium = MedNeXtMedium = lambda **kwargs: create_mednext("M", **kwargs)
MedNextL = MedNeXtL = MedNextLarge = MedNeXtLarge = lambda **kwargs: create_mednext("L", **kwargs)
