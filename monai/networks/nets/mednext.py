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
from dataclasses import dataclass

import torch
import torch.nn as nn

from monai.networks.blocks.mednext_block import MedNeXtBlock, MedNeXtDownBlock, MedNeXtOutBlock, MedNeXtUpBlock


@dataclass
class MedNextConfig:
    init_filters: int
    in_channels: int
    out_channels: int
    encoder_expansion_ratio: Sequence[int]
    decoder_expansion_ratio: Sequence[int]
    bottleneck_expansion_ratio: int
    blocks_down: Sequence[int]
    blocks_bottleneck: int
    blocks_up: Sequence[int]
    kernel_size: int


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
        config: MedNextConfig,
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

        self.config = config

        encoder_kernel_size = decoder_kernel_size = self.config.kernel_size

        self.stem = nn.Conv2d(self.config.in_channels, self.config.init_filters, kernel_size=1)

        enc_stages = []
        down_blocks = []

        for i, num_blocks in enumerate(self.config.blocks_down):
            enc_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=self.config.init_filters * (2**i),
                            out_channels=self.config.init_filters * (2**i),
                            expansion_ratio=self.config.encoder_expansion_ratio[i],
                            kernel_size=encoder_kernel_size,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

            down_blocks.append(
                MedNeXtDownBlock(
                    in_channels=self.config.init_filters * (2**i),
                    out_channels=self.config.init_filters * (2 ** (i + 1)),
                    expansion_ratio=self.config.encoder_expansion_ratio[i],
                    kernel_size=encoder_kernel_size,
                )
            )

        self.enc_stages = nn.ModuleList(enc_stages)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=self.config.init_filters * (2 ** len(self.config.blocks_down)),
                    out_channels=self.config.init_filters * (2 ** len(self.config.blocks_down)),
                    expansion_ratio=self.config.bottleneck_expansion_ratio,
                    kernel_size=decoder_kernel_size,
                )
                for _ in range(self.config.blocks_bottleneck)
            ]
        )

        up_blocks = []
        dec_stages = []
        for i, num_blocks in enumerate(self.config.blocks_up):
            up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=self.config.init_filters * (2 ** (len(self.config.blocks_up) - i)),
                    out_channels=self.config.init_filters * (2 ** (len(self.config.blocks_up) - i - 1)),
                    expansion_ratio=self.config.decoder_expansion_ratio[i],
                    kernel_size=decoder_kernel_size,
                )
            )

            dec_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=self.config.init_filters * (2 ** (len(self.config.blocks_up) - i - 1)),
                            out_channels=self.config.init_filters * (2 ** (len(self.config.blocks_up) - i - 1)),
                            expansion_ratio=self.config.decoder_expansion_ratio[i],
                            kernel_size=decoder_kernel_size,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.out_0 = MedNeXtOutBlock(
            in_channels=self.config.init_filters,
            n_classes=self.config.out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            torch.Tensor or Tuple[torch.Tensor, ...]: Output tensor(s).
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

        # Decoder forward pass with skip connections
        for i, (up_block, dec_stage) in enumerate(zip(self.up_blocks, self.dec_stages)):
            x = up_block(x)
            x = x + enc_outputs[-(i + 1)]
            x = dec_stage(x)

        # Final output block
        x = self.out_0(x)

        return x


COMMON_ARGS = {
    "init_filters": 32,
    "in_channels": 4,
    "out_channels": 3,
    "kernel_size": 3,
}

MED_NEXT_SMALL = {
    **COMMON_ARGS,
    "encoder_expansion_ratio": (2, 2, 2, 2),
    "decoder_expansion_ratio": (2, 2, 2, 2),
    "bottleneck_expansion_ratio": 2,
    "blocks_down": (2, 2, 2, 2),
    "blocks_bottleneck": 2,
    "blocks_up": (2, 2, 2, 2),
}

MED_NEXT_BASE = {
    **COMMON_ARGS,
    "encoder_expansion_ratio": (2, 3, 4, 4),
    "decoder_expansion_ratio": (4, 4, 3, 2),
    "bottleneck_expansion_ratio": 4,
    "blocks_down": (2, 2, 2, 2),
    "blocks_bottleneck": 2,
    "blocks_up": (2, 2, 2, 2),
}

MED_NEXT_MEDIUM = {
    **COMMON_ARGS,
    "encoder_expansion_ratio": (2, 3, 4, 4),
    "decoder_expansion_ratio": (4, 4, 3, 2),
    "bottleneck_expansion_ratio": 4,
    "blocks_down": (3, 4, 4, 4),
    "blocks_bottleneck": 4,
    "blocks_up": (4, 4, 4, 3),
}

MED_NEXT_LARGE = {
    **COMMON_ARGS,
    "encoder_expansion_ratio": (3, 4, 8, 8),
    "decoder_expansion_ratio": (8, 8, 4, 3),
    "bottleneck_expansion_ratio": 8,
    "blocks_down": (3, 4, 8, 8),
    "blocks_bottleneck": 8,
    "blocks_up": (8, 8, 4, 3),
}


def create_mednext(
    variant: str = "S",
    in_channels: int | None = None,
    out_channels: int | None = None,
) -> MedNeXt:
    """
    Create a MedNeXt model.

    Args:
        variant: the variant of the model to create. Defaults to "S".
        in_channels: number of input channels for the network. Defaults to 4.
        out_channels: number of output channels for the network. Defaults to 3.

    Returns:
        MedNeXt: the created model.
    """

    if variant == "S":
        config = MedNextConfig(**MED_NEXT_SMALL)  # type: ignore
    elif variant == "B":
        config = MedNextConfig(**MED_NEXT_BASE)  # type: ignore
    elif variant == "M":
        config = MedNextConfig(**MED_NEXT_MEDIUM)  # type: ignore
    elif variant == "L":
        config = MedNextConfig(**MED_NEXT_LARGE)  # type: ignore
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if in_channels:
        config.in_channels = in_channels
    if out_channels:
        config.out_channels = out_channels

    return MedNeXt(config)
