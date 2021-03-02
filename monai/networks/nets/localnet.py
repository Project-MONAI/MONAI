from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks.localnet_block import (
    get_conv_block, get_deconv_block,
)
from monai.networks.nets import RegUNet


class AdditiveUpSampleBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
    ):
        super(AdditiveUpSampleBlock, self).__init__()
        self.deconv = get_deconv_block(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = (size * 2 for size in x.shape[2:])
        deconved = self.deconv(x)
        resized = F.interpolate(x, output_size)
        resized = torch.sum(
            torch.stack(
                resized.split(split_size=resized.shape[1] // 2, dim=1),
                dim=-1),
            dim=-1
        )
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
            encode_kernel_sizes=[7] + [3] * max(extract_levels)
        )

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return get_conv_block(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def build_up_sampling_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        if self._use_additive_upsampling:
            return AdditiveUpSampleBlock(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels
            )

        return get_deconv_block(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels
        )
