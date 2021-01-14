from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks.localnet_block import (
    LocalNetDownSampleBlock,
    LocalNetFeatureExtractorBlock,
    LocalNetUpSampleBlock,
    get_conv_block,
)


class LocalNet(nn.Module):
    """
    Reimplementation of LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channel_initial: int,
        extract_levels: List[int],
        out_activation: Optional[Union[Tuple, str]],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_channel_initial: number of initial channels,
            extract_levels: number of extraction levels,
            out_activation: activation to use at end layer,
        """
        super(LocalNet, self).__init__()
        self.extract_levels = extract_levels
        self.extract_max_level = max(self.extract_levels)  # E
        self.extract_min_level = min(self.extract_levels)  # D

        num_channels = [
            num_channel_initial * (2 ** level) for level in range(self.extract_max_level + 1)
        ]  # level 0 to E

        self.downsample_blocks = nn.ModuleList(
            [
                LocalNetDownSampleBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else num_channels[i - 1],
                    out_channels=num_channels[i],
                    kernel_size=7 if i == 0 else 3,
                )
                for i in range(self.extract_max_level)
            ]
        )  # level 0 to self.extract_max_level - 1
        self.conv3d_block = get_conv_block(
            spatial_dims=spatial_dims, in_channels=num_channels[-2], out_channels=num_channels[-1]
        )  # self.extract_max_level

        self.upsample_blocks = nn.ModuleList(
            [
                LocalNetUpSampleBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[level + 1],
                    out_channels=num_channels[level],
                )
                for level in range(self.extract_max_level - 1, self.extract_min_level - 1, -1)
            ]
        )  # self.extract_max_level - 1 to self.extract_min_level

        self.extract_layers = nn.ModuleList(
            [
                # if kernels are not initialized by zeros, with init NN, extract may be too large
                LocalNetFeatureExtractorBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[level],
                    out_channels=out_channels,
                    act=out_activation,
                )
                for level in self.extract_levels
            ]
        )

    def forward(self, x) -> torch.Tensor:
        image_size = x.shape[2:]
        for size in image_size:
            if size % (2 ** self.extract_max_level) != 0:
                raise ValueError(
                    f"given extract_max_level {self.extract_max_level}, "
                    f"all input spatial dimension must be devidable by {2 ** self.extract_max_level}, "
                    f"got input of size {image_size}"
                )
        mid_features = []  # 0 -> self.extract_max_level - 1
        for downsample_block in self.downsample_blocks:
            x, mid = downsample_block(x)
            mid_features.append(mid)
        x = self.conv3d_block(x)  # self.extract_max_level

        decoded_features = [x]
        for idx, upsample_block in enumerate(self.upsample_blocks):
            x = upsample_block(x, mid_features[-idx - 1])
            decoded_features.append(x)  # self.extract_max_level -> self.extract_min_level

        output = torch.mean(
            torch.stack(
                [
                    F.interpolate(
                        extract_layer(decoded_features[self.extract_max_level - self.extract_levels[idx]]),
                        size=image_size,
                    )
                    for idx, extract_layer in enumerate(self.extract_layers)
                ],
                dim=-1,
            ),
            dim=-1,
        )
        return output
