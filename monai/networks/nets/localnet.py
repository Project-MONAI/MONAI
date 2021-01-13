from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks.localnet_block import LocalNetDownSampleBlock, get_conv_block, LocalNetUpSampleBlock, \
    ExtractBlock


class LocalNet(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_channel_initial: int,
            extract_levels: List[int],
            out_kernel_initializer: str,
            out_activation: Optional[Union[Tuple, str]],
            control_points: (tuple, None) = None,
            **kwargs,
    ):
        super(LocalNet, self).__init__()
        self.extract_levels = extract_levels
        self.extract_max_level = max(self.extract_levels)  # E
        self.extract_min_level = min(self.extract_levels)  # D

        num_channels = [
            num_channel_initial * (2 ** level)
            for level in range(self.extract_max_level + 1)
        ]  # level 0 to E

        self.downsample_blocks = nn.ModuleList([
            LocalNetDownSampleBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels if i == 0 else num_channels[i - 1],
                out_channels=num_channels[i],
                kernel_size=7 if i == 0 else 3
            )
            for i in range(self.extract_max_level)
        ])  # level 0 to E-1
        self.conv3d_block = get_conv_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-2],
            out_channels=num_channels[-1]
        )  # level E

        self.upsample_blocks = nn.ModuleList([
            LocalNetUpSampleBlock(
                spatial_dims=spatial_dims,
                in_channels=num_channels[level + 1],
                out_channels=num_channels[level],
            )
            for level in range(
                self.extract_max_level - 1, self.extract_min_level - 1, -1
            )
        ])  # level D to E-1

        self.extract_layers = nn.ModuleList([
            # if kernels are not initialized by zeros, with init NN, extract may be too large
            ExtractBlock(
                spatial_dims=spatial_dims,
                in_channels=num_channels[level],
                out_channels=out_channels,
                kernel_initializer=out_kernel_initializer,
                act=out_activation
            )
            for level in self.extract_levels
        ])

    def forward(self, x):
        image_size = x.shape[2:]
        for size in image_size:
            if size % (2 ** self.extract_max_level) != 0:
                raise ValueError(
                    f"given extract_max_level {self.extract_max_level}, "
                    f"all input spatial dimension must be devidable by {2 ** self.extract_max_level}, "
                    f"got input of size {image_size}")
        image_size = tuple(image_size)
        encoded = []
        h_in = x
        for level in range(self.extract_max_level):
            h_in, h_channel = self.downsample_blocks[level](h_in)
            encoded.append(h_channel)
        h_bottom = self.conv3d_block(h_in)

        decoded = [h_bottom]
        # level E-1 to D
        for idx, level in enumerate(range(self.extract_max_level - 1, self.extract_min_level - 1, -1)):
            h_bottom = self.upsample_blocks[idx](h_bottom, encoded[level])
            decoded.append(h_bottom)

        output = torch.mean(
            torch.stack(
                [
                    F.interpolate(
                        self.extract_layers[idx](decoded[self.extract_max_level - level]),
                        size=image_size
                    )
                    for idx, level in enumerate(self.extract_levels)
                ],
                dim=-1
            ),
            dim=-1
        )
        return output


if __name__ == '__main__':
    input = torch.rand((2, 2, 32, 32, 32))
    model = LocalNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        num_channel_initial=32,
        extract_levels=[0, 1, 2, 3],
        out_kernel_initializer="zeros",
        out_activation=None,
    )
    out = model(input)
    print(out.shape)



