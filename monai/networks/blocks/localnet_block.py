from typing import Union, Sequence, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks import Convolution, get_padding
from monai.networks.layers.factories import batch_factory, maxpooling_factory


initializer_dict = {
    "zeros": nn.init.zeros_,
}


def get_conv_block(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        act: Optional[Union[Tuple, str]] = "RELU",
        norm: Optional[Union[Tuple, str]] = "BATCH",
):
    padding = get_padding(kernel_size, stride=1)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        bias=False,
        conv_only=False,
        padding=padding,
    )


def get_conv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
):
    padding = get_padding(kernel_size, stride=1)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        bias=False,
        conv_only=True,
        padding=padding,
    )


def get_deconv_block(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
):
    return Convolution(
        dimensions=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        act="RELU",
        norm="BATCH",
        bias=False,
        is_transposed=True,
        padding=1,
        output_padding=1,
    )


class ResidualBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
    ):
        super(ResidualBlock, self).__init__()
        if in_channels != out_channels:
            raise ValueError(
                f"expecting in_channels == out_channels, "
                f"got in_channels={in_channels}, out_channels={out_channels}")
        self.conv_block = get_conv_block(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.conv = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        self.norm = batch_factory(spatial_dims)(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(
            self.norm(
                self.conv(
                    self.conv_block(x)
                )
            ) + x
        )


class LocalNetResidualBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
    ):
        super(LocalNetResidualBlock, self).__init__()
        # if in_channels != out_channels:
        #     raise ValueError(
        #         f"expecting in_channels == out_channels, "
        #         f"got in_channels={in_channels}, out_channels={out_channels}")
        self.conv_layer = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.norm = batch_factory(spatial_dims)(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, mid):
        return self.relu(
            self.norm(
                self.conv_layer(x)
            ) + mid
        )


class LocalNetDownSampleBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
    ):
        super(LocalNetDownSampleBlock, self).__init__()
        self.conv_block = get_conv_block(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        self.residual_block = ResidualBlock(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        self.max_pool = maxpooling_factory(spatial_dims)(
            kernel_size=2,
        )

    def forward(self, x):
        x = self.conv_block(x)
        mid = self.residual_block(x)
        x = self.max_pool(mid)
        return x, mid


class LocalNetUpSampleBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
    ):
        super(LocalNetUpSampleBlock, self).__init__()
        self.deconv_block = get_deconv_block(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.conv_block = get_conv_block(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.residual_block = LocalNetResidualBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        if in_channels / out_channels != 2:
            raise ValueError(
                f"expecting in_channels == 2 * out_channels, "
                f"got in_channels={in_channels}, out_channels={out_channels}"
            )
        self.out_channels = out_channels

    def addictive_upsampling(self, x):
        size = torch.Size(torch.tensor(x.shape[2:]) * 2)
        x = F.interpolate(x, size)
        # [(batch, out_channels, ...), (batch, out_channels, ...)]
        x = x.split(split_size=int(self.out_channels), dim=1)
        # (batch, out_channels, ...)
        x = torch.sum(
            torch.stack(x, dim=-1),
            dim=-1
        )
        return x

    def forward(self, x, mid):
        h0 = self.deconv_block(x) + self.addictive_upsampling(x)
        r1 = h0 + mid
        r2 = self.conv_block(h0)
        return self.residual_block(r2, r1)


class ExtractBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            act: Optional[Union[Tuple, str]] = "RELU",
            kernel_initializer: Optional[Union[Tuple, str]] = None,
    ):
        super(ExtractBlock, self).__init__()
        self.conv_block = get_conv_block(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            act=act,
            norm=None
        )
        if kernel_initializer:
            initializer_dict[kernel_initializer](self.conv_block.conv.weight)

    def forward(self, x):
        x = self.conv_block(x)
        return x