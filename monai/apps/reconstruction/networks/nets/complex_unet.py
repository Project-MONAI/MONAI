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

from typing import Optional, Sequence, Union

import torch.nn as nn
from torch import Tensor

from monai.apps.reconstruction.networks.nets.utils import (
    complex_normalize,
    divisible_pad_t,
    inverse_divisible_pad_t,
    reshape_channel_complex_to_last_dim,
    reshape_complex_to_channel_dim,
)
from monai.networks.nets.basic_unet import BasicUNet


class ComplexUnet(nn.Module):
    """
    This variant of U-Net handles complex-value input/output. It can be
    used as a model to learn sensitivity maps in multi-coil MRI data. It is
    built based on :py:class:`monai.networks.nets.BasicUNet` by default but the user
    can input their convolutional model as well.
    ComplexUnet also applies default normalization to the input which makes it more stable to train.

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
        pad_factor: an integer denoting the number which each padded dimension will be divisible to.
            For example, 16 means each dimension will be divisible by 16 after padding
        conv_net: the learning model used inside the ComplexUnet. The default
            is :py:class:`monai.networks.nets.basic_unet`. The only requirement on the model is to
            have 2 as input and output number of channels.
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
        pad_factor: int = 16,
        conv_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        if conv_net is None:
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
        else:
            # assume the first layer is convolutional and
            # check whether in_channels == 2
            params = [p.shape for p in conv_net.parameters()]
            if params[0][1] != 2:
                raise ValueError(f"in_channels should be 2 but it's {params[0][1]}.")
            self.unet = conv_net

        self.pad_factor = pad_factor

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

        Returns:
            output of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data
        """
        # suppose the input is 2D, the comment in front of each operator below shows the shape after that operator
        x = reshape_complex_to_channel_dim(x)  # x will be of shape (B,C*2,H,W)
        x, mean, std = complex_normalize(x)  # x will be of shape (B,C*2,H,W)
        # pad input
        x, padding_sizes = divisible_pad_t(
            x, k=self.pad_factor
        )  # x will be of shape (B,C*2,H',W') where H' and W' are for after padding

        x = self.unet(x)
        # inverse padding
        x = inverse_divisible_pad_t(x, padding_sizes)  # x will be of shape (B,C*2,H,W)

        x = x * std + mean
        x = reshape_channel_complex_to_last_dim(x)  # x will be of shape (B,C,H,W,2)
        return x
