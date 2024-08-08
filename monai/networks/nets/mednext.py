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

import torch
import torch.nn as nn

from monai.networks.blocks import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock, OutBlock


class MedNeXt(nn.Module):
    """
    MedNeXt model class from paper: https://arxiv.org/pdf/2303.09975

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        enc_exp_r: expansion ratio for encoder blocks. Defaults to 2.
        dec_exp_r: expansion ratio for decoder blocks. Defaults to 2.
        bottlenec_exp_r: expansion ratio for bottleneck blocks. Defaults to 2.
        kernel_size: kernel size for convolutions. Defaults to 7.
        deep_supervision: whether to use deep supervision. Defaults to False.
        do_res: whether to use residual connections. Defaults to False.
        do_res_up_down: whether to use residual connections in up and down blocks. Defaults to False.
        blocks_down: number of blocks in each encoder stage. Defaults to [2, 2, 2, 2].
        blocks_bottleneck: number of blocks in bottleneck stage. Defaults to 2.
        blocks_up: number of blocks in each decoder stage. Defaults to [2, 2, 2, 2].
        norm_type: type of normalization layer. Defaults to 'group'.
        grn: whether to use Global Response Normalization (GRN). Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 2,
        enc_exp_r: int = 2,
        dec_exp_r: int = 2,
        bottlenec_exp_r: int = 2,
        kernel_size: int = 7,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        blocks_down: list = [2, 2, 2, 2],
        blocks_bottleneck: int = 2,
        blocks_up: list = [2, 2, 2, 2],
        norm_type: str = "group",
        grn: bool = False,
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

        if isinstance(enc_exp_r, int):
            enc_exp_r = [enc_exp_r] * len(blocks_down)

        if isinstance(dec_exp_r, int):
            dec_exp_r = [dec_exp_r] * len(blocks_up)

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
                            exp_r=enc_exp_r[i],
                            kernel_size=enc_kernel_size,
                            do_res=do_res,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            grn=grn,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

            down_blocks.append(
                MedNeXtDownBlock(
                    in_channels=init_filters * (2**i),
                    out_channels=init_filters * (2 ** (i + 1)),
                    exp_r=enc_exp_r[i],
                    kernel_size=enc_kernel_size,
                    do_res=do_res_up_down,
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
                    exp_r=bottlenec_exp_r,
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    grn=grn,
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
                    exp_r=dec_exp_r[i],
                    kernel_size=dec_kernel_size,
                    do_res=do_res_up_down,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    grn=grn,
                )
            )

            dec_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            out_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            exp_r=dec_exp_r[i],
                            kernel_size=dec_kernel_size,
                            do_res=do_res,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            grn=grn,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.out_0 = OutBlock(in_channels=init_filters, n_classes=out_channels, dim=spatial_dims_str)

        if deep_supervision:
            out_blocks = [
                OutBlock(in_channels=init_filters * (2**i), n_classes=out_channels, dim=spatial_dims_str)
                for i in range(1, len(blocks_up) + 1)
            ]

            out_blocks.reverse()
            self.out_blocks = nn.ModuleList(out_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
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
            torch.Tensor or list[torch.Tensor]: Output tensor(s).
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
