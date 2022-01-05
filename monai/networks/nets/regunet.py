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
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks.regunet_block import (
    RegistrationDownSampleBlock,
    RegistrationExtractionBlock,
    RegistrationResidualConvBlock,
    get_conv_block,
    get_deconv_block,
)

__all__ = ["RegUNet", "AffineHead", "GlobalNet", "LocalNet"]


class RegUNet(nn.Module):
    """
    Class that implements an adapted UNet. This class also serve as the parent class of LocalNet and GlobalNet

    Reference:
        O. Ronneberger, P. Fischer, and T. Brox,
        “U-net: Convolutional networks for biomedical image segmentation,”,
        Lecture Notes in Computer Science, 2015, vol. 9351, pp. 234–241.
        https://arxiv.org/abs/1505.04597

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        extract_levels: Optional[Tuple[int]] = None,
        pooling: bool = True,
        concat_skip: bool = False,
        encode_kernel_sizes: Union[int, List[int]] = 3,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            depth: input is at level 0, bottom is at level depth.
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
            encode_kernel_sizes: kernel size for down-sampling
        """
        super().__init__()
        if not extract_levels:
            extract_levels = (depth,)
        if max(extract_levels) != depth:
            raise AssertionError

        # save parameters
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channel_initial = num_channel_initial
        self.depth = depth
        self.out_kernel_initializer = out_kernel_initializer
        self.out_activation = out_activation
        self.out_channels = out_channels
        self.extract_levels = extract_levels
        self.pooling = pooling
        self.concat_skip = concat_skip

        if isinstance(encode_kernel_sizes, int):
            encode_kernel_sizes = [encode_kernel_sizes] * (self.depth + 1)
        if len(encode_kernel_sizes) != self.depth + 1:
            raise AssertionError
        self.encode_kernel_sizes: List[int] = encode_kernel_sizes

        self.num_channels = [self.num_channel_initial * (2 ** d) for d in range(self.depth + 1)]
        self.min_extract_level = min(self.extract_levels)

        # init layers
        # all lists start with d = 0
        self.encode_convs = None
        self.encode_pools = None
        self.bottom_block = None
        self.decode_deconvs = None
        self.decode_convs = None
        self.output_block = None

        # build layers
        self.build_layers()

    def build_layers(self):
        self.build_encode_layers()
        self.build_decode_layers()

    def build_encode_layers(self):
        # encoding / down-sampling
        self.encode_convs = nn.ModuleList(
            [
                self.build_conv_block(
                    in_channels=self.in_channels if d == 0 else self.num_channels[d - 1],
                    out_channels=self.num_channels[d],
                    kernel_size=self.encode_kernel_sizes[d],
                )
                for d in range(self.depth)
            ]
        )
        self.encode_pools = nn.ModuleList(
            [self.build_down_sampling_block(channels=self.num_channels[d]) for d in range(self.depth)]
        )
        self.bottom_block = self.build_bottom_block(
            in_channels=self.num_channels[-2], out_channels=self.num_channels[-1]
        )

    def build_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            get_conv_block(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            RegistrationResidualConvBlock(
                spatial_dims=self.spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def build_down_sampling_block(self, channels: int):
        return RegistrationDownSampleBlock(spatial_dims=self.spatial_dims, channels=channels, pooling=self.pooling)

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return nn.Sequential(
            get_conv_block(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            RegistrationResidualConvBlock(
                spatial_dims=self.spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def build_decode_layers(self):
        # decoding / up-sampling
        # [depth - 1, depth - 2, ..., min_extract_level]
        self.decode_deconvs = nn.ModuleList(
            [
                self.build_up_sampling_block(in_channels=self.num_channels[d + 1], out_channels=self.num_channels[d])
                for d in range(self.depth - 1, self.min_extract_level - 1, -1)
            ]
        )
        self.decode_convs = nn.ModuleList(
            [
                self.build_conv_block(
                    in_channels=(2 * self.num_channels[d] if self.concat_skip else self.num_channels[d]),
                    out_channels=self.num_channels[d],
                    kernel_size=3,
                )
                for d in range(self.depth - 1, self.min_extract_level - 1, -1)
            ]
        )

        # extraction
        self.output_block = self.build_output_block()

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels)

    def build_output_block(self) -> nn.Module:
        return RegistrationExtractionBlock(
            spatial_dims=self.spatial_dims,
            extract_levels=self.extract_levels,
            num_channels=self.num_channels,
            out_channels=self.out_channels,
            kernel_initializer=self.out_kernel_initializer,
            activation=self.out_activation,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])

        Returns:
            Tensor in shape (batch, ``out_channels``, insize_1, insize_2, [insize_3]), with the same spatial size as ``x``
        """
        image_size = x.shape[2:]
        skips = []  # [0, ..., depth - 1]
        encoded = x
        for encode_conv, encode_pool in zip(self.encode_convs, self.encode_pools):
            skip = encode_conv(encoded)
            encoded = encode_pool(skip)
            skips.append(skip)
        decoded = self.bottom_block(encoded)

        outs = [decoded]

        # [depth - 1, ..., min_extract_level]
        for i, (decode_deconv, decode_conv) in enumerate(zip(self.decode_deconvs, self.decode_convs)):
            # [depth - 1, depth - 2, ..., min_extract_level]
            decoded = decode_deconv(decoded)
            if self.concat_skip:
                decoded = torch.cat([decoded, skips[-i - 1]], dim=1)
            else:
                decoded = decoded + skips[-i - 1]
            decoded = decode_conv(decoded)
            outs.append(decoded)

        out = self.output_block(outs, image_size=image_size)
        return out


class AffineHead(nn.Module):
    def __init__(self, spatial_dims: int, image_size: List[int], decode_size: List[int], in_channels: int):
        super().__init__()
        self.spatial_dims = spatial_dims
        if spatial_dims == 2:
            in_features = in_channels * decode_size[0] * decode_size[1]
            out_features = 6
            out_init = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        elif spatial_dims == 3:
            in_features = in_channels * decode_size[0] * decode_size[1] * decode_size[2]
            out_features = 12
            out_init = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)
        else:
            raise ValueError(f"only support 2D/3D operation, got spatial_dims={spatial_dims}")

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.grid = self.get_reference_grid(image_size)  # (spatial_dims, ...)

        # init weight/bias
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(out_init)

    @staticmethod
    def get_reference_grid(image_size: Union[Tuple[int], List[int]]) -> torch.Tensor:
        mesh_points = [torch.arange(0, dim) for dim in image_size]
        grid = torch.stack(torch.meshgrid(*mesh_points), dim=0)  # (spatial_dims, ...)
        return grid.to(dtype=torch.float)

    def affine_transform(self, theta: torch.Tensor):
        # (spatial_dims, ...) -> (spatial_dims + 1, ...)
        grid_padded = torch.cat([self.grid, torch.ones_like(self.grid[:1])])

        # grid_warped[b,p,...] = sum_over_q(grid_padded[q,...] * theta[b,p,q]
        if self.spatial_dims == 2:
            grid_warped = torch.einsum("qij,bpq->bpij", grid_padded, theta.reshape(-1, 2, 3))
        elif self.spatial_dims == 3:
            grid_warped = torch.einsum("qijk,bpq->bpijk", grid_padded, theta.reshape(-1, 3, 4))
        else:
            raise ValueError(f"do not support spatial_dims={self.spatial_dims}")
        return grid_warped

    def forward(self, x: List[torch.Tensor], image_size: List[int]) -> torch.Tensor:
        f = x[0]
        self.grid = self.grid.to(device=f.device)
        theta = self.fc(f.reshape(f.shape[0], -1))
        out: torch.Tensor = self.affine_transform(theta) - self.grid
        return out


class GlobalNet(RegUNet):
    """
    Build GlobalNet for image registration.

    Reference:
        Hu, Yipeng, et al.
        "Label-driven weakly-supervised learning
        for multimodal deformable image registration,"
        https://arxiv.org/abs/1711.01666
    """

    def __init__(
        self,
        image_size: List[int],
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        pooling: bool = True,
        concat_skip: bool = False,
        encode_kernel_sizes: Union[int, List[int]] = 3,
    ):
        for size in image_size:
            if size % (2 ** depth) != 0:
                raise ValueError(
                    f"given depth {depth}, "
                    f"all input spatial dimension must be divisible by {2 ** depth}, "
                    f"got input of size {image_size}"
                )
        self.image_size = image_size
        self.decode_size = [size // (2 ** depth) for size in image_size]
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=depth,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=spatial_dims,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=encode_kernel_sizes,
        )

    def build_output_block(self):
        return AffineHead(
            spatial_dims=self.spatial_dims,
            image_size=self.image_size,
            decode_size=self.decode_size,
            in_channels=self.num_channels[-1],
        )


class AdditiveUpSampleBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = get_deconv_block(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = (size * 2 for size in x.shape[2:])
        deconved = self.deconv(x)
        resized = F.interpolate(x, output_size)
        resized = torch.sum(torch.stack(resized.split(split_size=resized.shape[1] // 2, dim=1), dim=-1), dim=-1)
        out: torch.Tensor = deconved + resized
        return out


class LocalNet(RegUNet):
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
        num_channel_initial: int,
        extract_levels: Tuple[int],
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        pooling: bool = True,
        concat_skip: bool = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
        """
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=max(extract_levels),
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=[7] + [3] * max(extract_levels),
        )

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return get_conv_block(
            spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self._use_additive_upsampling:
            return AdditiveUpSampleBlock(
                spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels
            )

        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels)
