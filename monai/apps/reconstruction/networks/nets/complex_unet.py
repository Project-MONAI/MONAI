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

import math
from typing import Sequence, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import monai


class ComplexUnet(nn.Module):
    """
    This variant of U-Net handles complex-value input/output. It can be
    used as a model to learn sensitivity maps in multi-coil MRI data. It is
    built based on :py:class:`monai.networks.nets.BasicUnet`.
    It also applies default normalization to the input which makes it more stable to train.

    The data being a (complex) 2-channel tensor is a requirement for using this model.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        spatial_dims: number of spatial dimensions.
        features: six integers as numbers of features. denotes number of channels in each layer.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
        dropout: dropout ratio. Defaults to 0.0.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        self.unet = monai.networks.nets.basic_unet.BasicUnet(
            spatial_dims=spatial_dims,
            in_channels=2,
            out_channels=2,
            features=features,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
        )

    def complex_to_chan_dim(self, x: Tensor) -> Tensor:
        """
        Swaps the complex dimension with the channel dimension so that the network treats real/imaginary
        parts as two separate channels.

        Args:
            x: input of shape (B,C,H,W,2)

        Returns:
            output of shape (B,C*2,H,W)
        """
        b, c, h, w, two = x.shape
        if x.shape[-1] != 2:
            raise ValueError(f"last dim must be 2, but x.shape[-1] is {x.shape[-1]}.")
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: Tensor) -> Tensor:
        """
        Swaps the complex dimension with the channel dimension so that the network output has 2 as its last dimension

        Args:
            x: input of shape (B,C*2,H,W)

        Returns:
            output of shape (B,C,H,W,2)
        """
        b, c2, h, w = x.shape  # c2 means c*2
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

    def normalize(self, x: Tensor) -> Sequence:
        """
        Performs group mean-std normalization. To see what "group" means, mean of
        an input of shape (B,C,H,W) will be (B,).

        Args:
            x: input of shape (B,C,H,W)

        Returns:
            A tuple containing
                (1) normalized output of shape (B,C,H,W)
                (2) mean
                (3) std
        """
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2, unbiased=False).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def reverse_normalize(self, x: Tensor, mean: float, std: float) -> Tensor:
        """
        Reverses the normalization done by norm

        Args:
            x: input of shape (B,C,H,W)
            mean: mean before normalization
            std: std before normalization

        Returns:
            denormalized output of shape (B,C,H,W)
        """
        return x * std + mean

    def pad(self, x: Tensor) -> Sequence:
        """
        Pad input to feed into the network
        """

        def floor_ceil(n):
            return math.floor(n), math.ceil(n)

        b, c, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def reverse_pad(self, x: Tensor, h_pad: Sequence, w_pad: Sequence, h_mult: int, w_mult: int) -> Tensor:
        """
        De-pad network output to match its original shape
        """
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W,2)

        Returns:
            output of shape (B,C,H,W,2)
        """
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.normalize(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.reverse_pad(x, *pad_sizes)
        x = self.reverse_normalize(x, mean, std)
        x = self.chan_complex_to_last_dim(x)
        return x
