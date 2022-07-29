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

from monai.networks.nets.basic_unet import BasicUNet
from monai.apps.reconstruction.networks.nets.utils import complex_to_channel_dim, complex_normalize, pad, reverse_pad, reverse_complex_normalize, channel_complex_to_last_dim


class ComplexUnet(nn.Module):
    """
    This variant of U-Net handles complex-value input/output. It can be
    used as a model to learn sensitivity maps in multi-coil MRI data. It is
    built based on :py:class:`monai.networks.nets.BasicUNet`.
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
        self.unet = BasicUNet(
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

        Returns:
            output of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data
        """
        x = complex_to_channel_dim(x)
        x, mean, std = complex_normalize(x)
        x, pad_sizes = pad(x)
        x = self.unet(x)
        x = reverse_pad(x, pad_sizes)
        x = reverse_complex_normalize(x, mean, std)
        x = channel_complex_to_last_dim(x)
        return x
