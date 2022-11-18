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

import torch
import torch.nn as nn

from monai.networks.layers.factories import Act, Norm, split_args
from monai.networks.nets.regressor import Regressor

__all__ = ["Classifier", "Discriminator", "Critic"]


class Classifier(Regressor):
    """
    Defines a classification network from Regressor by specifying the output shape as a single dimensional tensor
    with size equal to the number of classes to predict. The final activation function can also be specified, eg.
    softmax or sigmoid.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        classes: integer stating the dimension of the final output tensor
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component
        last_act: name defining the last activation layer
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        classes: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        last_act: Optional[str] = None,
    ) -> None:
        super().__init__(in_shape, (classes,), channels, strides, kernel_size, num_res_units, act, norm, dropout, bias)

        if last_act is not None:
            last_act_name, last_act_args = split_args(last_act)
            last_act_type = Act[last_act_name]

            self.final.add_module("lastact", last_act_type(**last_act_args))


class Discriminator(Classifier):
    """
    Defines a discriminator network from Classifier with a single output value and sigmoid activation by default. This
    is meant for use with GANs or other applications requiring a generic discriminator network.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component
        last_act: name defining the last activation layer
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = 0.25,
        bias: bool = True,
        last_act=Act.SIGMOID,
    ) -> None:
        super().__init__(in_shape, 1, channels, strides, kernel_size, num_res_units, act, norm, dropout, bias, last_act)


class Critic(Classifier):
    """
    Defines a critic network from Classifier with a single output value and no final activation. The final layer is
    `nn.Flatten` instead of `nn.Linear`, the final result is computed as the mean over the first dimension. This is
    meant to be used with Wasserstein GANs.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = 0.25,
        bias: bool = True,
    ) -> None:
        super().__init__(in_shape, 1, channels, strides, kernel_size, num_res_units, act, norm, dropout, bias, None)

    def _get_final_layer(self, in_shape: Sequence[int]):
        return nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.final(x)
        x = x.mean(1)
        return x.view((x.shape[0], -1))
