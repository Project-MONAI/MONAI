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

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution
from monai.networks.blocks.spade_norm import SPADE
from monai.networks.layers import Act
from monai.networks.layers.utils import get_act_layer
from monai.utils.enums import StrEnum

__all__ = ["SPADENet"]


class UpsamplingModes(StrEnum):
    bicubic = "bicubic"
    nearest = "nearest"
    bilinear = "bilinear"


class SPADENetResBlock(nn.Module):
    """
    Creates a Residual Block with SPADE normalisation.

    Args:
        spatial_dims: number of spatial dimensions
        in_channels: number of input channels
        out_channels: number of output channels
        label_nc: number of semantic channels that will be taken into account in SPADE normalisation blocks
        spade_intermediate_channels: number of intermediate channels in the middle conv. layers in SPADE normalisation blocks
        norm: base normalisation type used on top of SPADE
        kernel_size: convolutional kernel size
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        label_nc: int,
        spade_intermediate_channels: int = 128,
        norm: str | tuple = "INSTANCE",
        act: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = min(self.in_channels, self.out_channels)
        self.learned_shortcut = self.in_channels != self.out_channels
        self.conv_0 = Convolution(
            spatial_dims=spatial_dims, in_channels=self.in_channels, out_channels=self.int_channels, act=None, norm=None
        )
        self.conv_1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.int_channels,
            out_channels=self.out_channels,
            act=None,
            norm=None,
        )
        self.activation = get_act_layer(act)
        self.norm_0 = SPADE(
            label_nc=label_nc,
            norm_nc=self.in_channels,
            kernel_size=kernel_size,
            spatial_dims=spatial_dims,
            hidden_channels=spade_intermediate_channels,
            norm=norm,
        )
        self.norm_1 = SPADE(
            label_nc=label_nc,
            norm_nc=self.int_channels,
            kernel_size=kernel_size,
            spatial_dims=spatial_dims,
            hidden_channels=spade_intermediate_channels,
            norm=norm,
        )

        if self.learned_shortcut:
            self.conv_s = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                act=None,
                norm=None,
                kernel_size=1,
            )
            self.norm_s = SPADE(
                label_nc=label_nc,
                norm_nc=self.in_channels,
                kernel_size=kernel_size,
                spatial_dims=spatial_dims,
                hidden_channels=spade_intermediate_channels,
                norm=norm,
            )

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.activation(self.norm_0(x, seg)))
        dx = self.conv_1(self.activation(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s


class SPADEEncoder(nn.Module):
    """
    Encoding branch of a VAE compatible with a SPADE-like generator

    Args:
        spatial_dims: number of spatial dimensions
        in_channels: number of input channels
        z_dim: latent space dimension of the VAE containing the image sytle information
        channels: number of output after each downsampling block
        input_shape: spatial input shape of the tensor, necessary to do the reshaping after the linear layers
        of the autoencoder (HxWx[D])
        kernel_size: convolutional kernel size
        norm: normalisation layer type
        act: activation type
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        z_dim: int,
        channels: Sequence[int],
        input_shape: Sequence[int],
        kernel_size: int = 3,
        norm: str | tuple = "INSTANCE",
        act: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.channels = channels
        if len(input_shape) != spatial_dims:
            raise ValueError("Length of parameter input shape must match spatial_dims; got %s" % (input_shape))
        for s_ind, s_ in enumerate(input_shape):
            if s_ / (2 ** len(channels)) != s_ // (2 ** len(channels)):
                raise ValueError(
                    "Each dimension of your input must be divisible by 2 ** (autoencoder depth)."
                    "The shape in position %d, %d is not divisible by %d. " % (s_ind, s_, len(channels))
                )
        self.input_shape = input_shape
        self.latent_spatial_shape = [s_ // (2 ** len(self.channels)) for s_ in self.input_shape]
        blocks = []
        ch_init = self.in_channels
        for _, ch_value in enumerate(channels):
            blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=ch_init,
                    out_channels=ch_value,
                    strides=2,
                    kernel_size=kernel_size,
                    norm=norm,
                    act=act,
                )
            )
            ch_init = ch_value

        self.blocks = nn.ModuleList(blocks)
        self.fc_mu = nn.Linear(
            in_features=np.prod(self.latent_spatial_shape) * self.channels[-1], out_features=self.z_dim
        )
        self.fc_var = nn.Linear(
            in_features=np.prod(self.latent_spatial_shape) * self.channels[-1], out_features=self.z_dim
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def encode(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return self.reparameterize(mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


class SPADEDecoder(nn.Module):
    """
    Decoder branch of a SPADE-like generator. It can be used independently, without an encoding branch,
    behaving like a GAN, or coupled to a SPADE encoder.

    Args:
        label_nc: number of semantic labels
        spatial_dims: number of spatial dimensions
        out_channels: number of output channels
        label_nc: number of semantic channels used for the SPADE normalisation blocks
        input_shape: spatial input shape of the tensor, necessary to do the reshaping after the linear layers
        channels: number of output after each downsampling block
        z_dim: latent space dimension of the VAE containing the image sytle information (None if encoder is not used)
        is_vae: whether the decoder is going to be coupled to an autoencoder or not (true: yes, false: no)
        spade_intermediate_channels: number of channels in the intermediate layers of the SPADE normalisation blocks
        norm: base normalisation type
        act:  activation layer type
        last_act: activation layer type for the last layer of the network (can differ from previous)
        kernel_size: convolutional kernel size
        upsampling_mode: upsampling mode (nearest, bilinear etc.)
    """

    def __init__(
        self,
        spatial_dims: int,
        out_channels: int,
        label_nc: int,
        input_shape: Sequence[int],
        channels: list[int],
        z_dim: int | None = None,
        is_vae: bool = True,
        spade_intermediate_channels: int = 128,
        norm: str | tuple = "INSTANCE",
        act: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        last_act: str | tuple | None = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        kernel_size: int = 3,
        upsampling_mode: str = UpsamplingModes.nearest.value,
    ):
        super().__init__()
        self.is_vae = is_vae
        self.out_channels = out_channels
        self.label_nc = label_nc
        self.num_channels = channels
        if len(input_shape) != spatial_dims:
            raise ValueError("Length of parameter input shape must match spatial_dims; got %s" % (input_shape))
        for s_ind, s_ in enumerate(input_shape):
            if s_ / (2 ** len(channels)) != s_ // (2 ** len(channels)):
                raise ValueError(
                    "Each dimension of your input must be divisible by 2 ** (autoencoder depth)."
                    "The shape in position %d, %d is not divisible by %d. " % (s_ind, s_, len(channels))
                )
        self.latent_spatial_shape = [s_ // (2 ** len(self.num_channels)) for s_ in input_shape]

        if not self.is_vae:
            self.conv_init = Convolution(
                spatial_dims=spatial_dims, in_channels=label_nc, out_channels=channels[0], kernel_size=kernel_size
            )
        elif self.is_vae and z_dim is None:
            raise ValueError(
                "If the network is used in VAE-GAN mode, parameter z_dim "
                "(number of latent channels in the VAE) must be populated."
            )
        else:
            self.fc = nn.Linear(z_dim, np.prod(self.latent_spatial_shape) * channels[0])

        self.z_dim = z_dim
        blocks = []
        channels.append(self.out_channels)
        self.upsampling = torch.nn.Upsample(scale_factor=2, mode=upsampling_mode)
        for ch_ind, ch_value in enumerate(channels[:-1]):
            blocks.append(
                SPADENetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=ch_value,
                    out_channels=channels[ch_ind + 1],
                    label_nc=label_nc,
                    spade_intermediate_channels=spade_intermediate_channels,
                    norm=norm,
                    kernel_size=kernel_size,
                    act=act,
                )
            )

        self.blocks = torch.nn.ModuleList(blocks)
        self.last_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[-1],
            out_channels=out_channels,
            padding=(kernel_size - 1) // 2,
            kernel_size=kernel_size,
            norm=None,
            act=last_act,
        )

    def forward(self, seg, z: torch.Tensor | None = None):
        """
        Args:
            seg: input BxCxHxW[xD] semantic map on which the output is conditioned on
            z: latent vector output by the encoder if self.is_vae is True. When is_vae is
            False, z is a random noise vector.

        Returns:

        """
        if not self.is_vae:
            x = F.interpolate(seg, size=tuple(self.latent_spatial_shape))
            x = self.conv_init(x)
        else:
            if (
                z is None and self.z_dim is not None
            ):  # Even though this network is a VAE (self.is_vae), you should be able to sample from noise as well.
                z = torch.randn(seg.size(0), self.z_dim, dtype=torch.float32, device=seg.get_device())
            x = self.fc(z)
            x = x.view(*[-1, self.num_channels[0]] + self.latent_spatial_shape)

        for res_block in self.blocks:
            x = res_block(x, seg)
            x = self.upsampling(x)

        x = self.last_conv(x)
        return x


class SPADENet(nn.Module):
    """
    SPADE Network, implemented based on the code by Park, T et al. in
    "Semantic Image Synthesis with Spatially-Adaptive Normalization"
    (https://github.com/NVlabs/SPADE)

    Args:
        spatial_dims: number of spatial dimensions
        in_channels: number of input channels
        out_channels: number of output channels
        label_nc: number of semantic channels used for the SPADE normalisation blocks
        input_shape:  spatial input shape of the tensor, necessary to do the reshaping after the linear layers
        channels: number of output after each downsampling block
        z_dim: latent space dimension of the VAE containing the image sytle information (None if encoder is not used)
        is_vae: whether the decoder is going to be coupled to an autoencoder (true) or not (false)
        spade_intermediate_channels: number of channels in the intermediate layers of the SPADE normalisation blocks
        norm: base normalisation type
        act: activation layer type
        last_act: activation layer type for the last layer of the network (can differ from previous)
        kernel_size: convolutional kernel size
        upsampling_mode: upsampling mode (nearest, bilinear etc.)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        label_nc: int,
        input_shape: Sequence[int],
        channels: list[int],
        z_dim: int | None = None,
        is_vae: bool = True,
        spade_intermediate_channels: int = 128,
        norm: str | tuple = "INSTANCE",
        act: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        last_act: str | tuple | None = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        kernel_size: int = 3,
        upsampling_mode: str = UpsamplingModes.nearest.value,
    ):
        super().__init__()
        self.is_vae = is_vae
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.label_nc = label_nc
        self.input_shape = input_shape

        if self.is_vae:
            if z_dim is None:
                ValueError("The latent space dimension mapped by parameter z_dim cannot be None is is_vae is True.")
            else:
                self.encoder = SPADEEncoder(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    z_dim=z_dim,
                    channels=channels,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    norm=norm,
                    act=act,
                )

        decoder_channels = channels
        decoder_channels.reverse()

        self.decoder = SPADEDecoder(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            label_nc=label_nc,
            input_shape=input_shape,
            channels=decoder_channels,
            z_dim=z_dim,
            is_vae=is_vae,
            spade_intermediate_channels=spade_intermediate_channels,
            norm=norm,
            act=act,
            last_act=last_act,
            kernel_size=kernel_size,
            upsampling_mode=upsampling_mode,
        )

    def forward(self, seg: torch.Tensor, x: torch.Tensor | None = None):
        z = None
        if self.is_vae:
            z_mu, z_logvar = self.encoder(x)
            z = self.encoder.reparameterize(z_mu, z_logvar)
            return self.decoder(seg, z), z_mu, z_logvar
        else:
            return (self.decoder(seg, z),)

    def encode(self, x: torch.Tensor):
        if self.is_vae:
            return self.encoder.encode(x)
        else:
            return None

    def decode(self, seg: torch.Tensor, z: torch.Tensor | None = None):
        return self.decoder(seg, z)
