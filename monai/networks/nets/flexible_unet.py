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

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from monai.networks.blocks import UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets import EfficientNetBNFeatures
from monai.networks.nets.basic_unet import UpCat
from monai.utils import InterpolateMode

__all__ = ["FlexibleUNet"]

encoder_feature_channel = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}


def _get_encoder_channels_by_backbone(backbone: str, in_channels: int = 3) -> tuple:
    """
    Get the encoder output channels by given backbone name.

    Args:
        backbone: name of backbone to generate features, can be from [efficientnet-b0, ..., efficientnet-b7].
        in_channels: channel of input tensor, default to 3.

    Returns:
        A tuple of output feature map channels' length .
    """
    encoder_channel_tuple = encoder_feature_channel[backbone]
    encoder_channel_list = [in_channels] + list(encoder_channel_tuple)
    encoder_channel = tuple(encoder_channel_list)
    return encoder_channel


class UNetDecoder(nn.Module):
    """
    UNet Decoder.
    This class refers to `segmentation_models.pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        encoder_channels: number of output channels for all feature maps in encoder.
            `len(encoder_channels)` should be no less than 2.
        decoder_channels: number of output channels for all feature maps in decoder.
            `len(decoder_channels)` should equal to `len(encoder_channels) - 1`.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        dropout: dropout ratio.
        bias: whether to have a bias term in convolution blocks in this decoder.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        pre_conv: a conv block applied before upsampling.
            Only used in the "nontrainable" or "pixelshuffle" mode.
        interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
            Only used in the "nontrainable" mode.
        align_corners: set the align_corners parameter for upsample. Defaults to True.
            Only used in the "nontrainable" mode.

    """

    def __init__(
        self,
        spatial_dims: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple],
        bias: bool,
        upsample: str,
        pre_conv: Optional[str],
        interp_mode: str,
        align_corners: Optional[bool],
    ):

        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("the length of `encoder_channels` should be no less than 2.")
        if len(decoder_channels) != len(encoder_channels) - 1:
            raise ValueError("`len(decoder_channels)` should equal to `len(encoder_channels) - 1`.")

        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1)
        halves.append(False)
        blocks = []
        for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves):
            blocks.append(
                UpCat(
                    spatial_dims=spatial_dims,
                    in_chns=in_chn,
                    cat_chns=skip_chn,
                    out_chns=out_chn,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    upsample=upsample,
                    pre_conv=pre_conv,
                    interp_mode=interp_mode,
                    align_corners=align_corners,
                    halves=halve,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: Sequence[torch.Tensor], skip_connect: int = 4):
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        x = features[0]
        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)

        return x


class SegmentationHead(nn.Sequential):
    """
    Segmentation head.
    This class refers to `segmentation_models.pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_.

    Args:
        dim: number of spatial dimensions.
        in_channels: number of input channels for the block.
        out_channels: number of output channels for the block.
        kernel_size: kernel size for the conv layer.
        act: activation type and arguments.
        scale_factor: multiplier for spatial size. Has to match input size if it is a tuple.

    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        act: Optional[Union[Tuple, str]] = None,
        scale_factor: float = 1.0,
    ):

        conv_layer = Conv[Conv.CONV, dim](
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        up_layer: nn.Module = nn.Identity()
        if scale_factor > 1.0:
            up_layer = UpSample(
                spatial_dims=dim,
                scale_factor=scale_factor,
                mode="nontrainable",
                pre_conv=None,
                interp_mode=InterpolateMode.LINEAR,
            )
        if act is not None:
            act_layer = get_act_layer(act)
        else:
            act_layer = nn.Identity()
        super().__init__(conv_layer, up_layer, act_layer)


class FlexibleUNet(nn.Module):
    """
    A flexible implementation of UNet-like encoder-decoder architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str,
        pretrained: bool = False,
        decoder_channels: Tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout: Union[float, tuple] = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        interp_mode: str = "nearest",
    ) -> None:
        """
        A flexible implement of UNet, in which the backbone/encoder can be replaced with
        any efficient network. Currently the input must have a 2 or 3 spatial dimension
        and the spatial size of each dimension must be a multiple of 32

        TODO(binliu@nvidia.com): Add more backbones/encoders to this class and make a general
        encoder-decoder structure. ETC:2022.09.01

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            backbone: name of backbones to initialize, only support efficientnet right now,
                can be from [efficientnet-b0,..., efficientnet-b8, efficientnet-l2].
            pretrained: whether to initialize pretrained ImageNet weights, only available
                for spatial_dims=2 and batch norm is used, default to False.
            decoder_channels: number of output channels for all feature maps in decoder.
                `len(decoder_channels)` should equal to `len(encoder_channels) - 1`,default
                to (256, 128, 64, 32, 16).
            spatial_dims: number of spatial dimensions, default to 2.
            norm: normalization type and arguments, default to ("batch", {"eps": 1e-3,
                "momentum": 0.1}).
            act: activation type and arguments, default to ("relu", {"inplace": True}).
            dropout: dropout ratio, default to 0.0.
            decoder_bias: whether to have a bias term in decoder's convolution blocks.
            upsample: upsampling mode, available options are``"deconv"``, ``"pixelshuffle"``,
                ``"nontrainable"``.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
        """
        super().__init__()

        if backbone not in encoder_feature_channel:
            raise ValueError(f"invalid model_name {backbone} found, must be one of {encoder_feature_channel.keys()}.")

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        if "ap" in backbone:
            adv_prop = True
        else:
            adv_prop = False

        self.backbone = backbone
        self.spatial_dims = spatial_dims
        model_name = backbone
        encoder_channels = _get_encoder_channels_by_backbone(backbone, in_channels)
        self.encoder = EfficientNetBNFeatures(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            spatial_dims=spatial_dims,
            norm=norm,
            adv_prop=adv_prop,
        )
        self.decoder = UNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=None,
            align_corners=None,
        )

        self.segmentation_head = SegmentationHead(
            dim=spatial_dims, in_channels=decoder_channels[-1], out_channels=out_channels, kernel_size=3, act=None
        )

        self.pool = Pool[Pool.ADAPTIVEMAX, 2](1)

    def forward(self, inputs: torch.Tensor):
        """
        Do a typical encoder-decoder-header inference.

        Args:
            inputs: input should have spatially N dimensions ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``,
                N is defined by `dimensions`.

        Returns:
            A torch Tensor of "raw" predictions in shape ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.

        """
        x = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out)
        x_seg = self.segmentation_head(decoder_out)

        return x_seg
