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

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Bottleneck, ConvConcatDenseBlock, Decoder, Encoder
from monai.networks.blocks import squeeze_and_excitation as se
from monai.networks.blocks.convolutions import ClassifierBlock
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnectionWithIdx
from monai.networks.layers.utils import get_dropout_layer, get_pool_layer
from monai.utils import optional_import
# Lazy import to avoid dependency
se1, flag = optional_import("squeeze_and_excitation")

__all__ = ["Quicknat"]


class Quicknat(nn.Module):
    """
    Model for "Quick segmentation of NeuroAnaTomy (QuickNAT) based on a deep fully convolutional neural network.
    Refer to: "QuickNAT: A Fully Convolutional Network for Quick and Accurate Segmentation of Neuroanatomy by
    Abhijit Guha Roya, Sailesh Conjetib, Nassir Navabb, Christian Wachingera"

    QuickNAT has an encoder/decoder like 2D F-CNN architecture with 4 encoders and 4 decoders separated by a bottleneck layer.
    The final layer is a classifier block with softmax.
    The architecture includes skip connections between all encoder and decoder blocks of the same spatial resolution,
    similar to the U-Net architecture.
    All Encoder and Decoder consist of three convolutional layers all with a Batch Normalization and ReLU.
    The first two convolutional layers are followed by a concatenation layer that concatenates
    the input feature map with outputs of the current and previous convolutional blocks.
    The kernel size of the first two convolutional layers is 5*5, the third convolutional layer has a kernel size of 1*1.

    Data in the encode path is downsampled using max pooling layers instead of upsamling like UNet and in the decode path
    upsampled using max un-pooling layers instead of transpose convolutions.
    The pooling is done at the beginning of the block and the unpool afterwards.
    The indices of the max pooling in the Encoder are forwarded through the layer to be available to the corresponding Decoder.

    The bottleneck block consists of a 5 * 5 convolutional layer and a batch normalization layer
    to separate the encoder and decoder part of the network,
    restricting information flow between the encoder and decoder.

    The output feature map from the last decoder block is passed to the classifier block,
    which is a convolutional layer with 1 * 1 kernel size that maps the input to an N channel feature map,
    where N is the number of segmentation classes.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Args:
        num_classes: number of classes to segmentate (output channels).
        num_channels: number of input channels.
        num_filters: number of output channels for each convolutional layer in a Dense Block.
        kernel_size: size of the kernel of each convolutional layer in a Dense Block.
        kernel_c: convolution kernel size of classifier block kernel.
        stride_convolution: convolution stride. Defaults to 1.
        pool: kernel size of the pooling layer,
        stride_pool: stride for the pooling layer.
        se_block: Squeeze and Excite block type to be included, defaults to None. Valid options : NONE, CSE, SSE, CSSE,
        droup_out: dropout ratio. Defaults to no dropout.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import QuickNAT

        # network with max pooling by a factor of 2 at each layer with no se_block.
        net = QuickNAT(
            num_classes=3,
            num_channels=1,
            num_filters=64,
            pool = 2,
            se_block = "None"
        )

    """

    def __init__(
        self,
        num_classes: int = 33,
        num_channels: int = 1,
        num_filters: int = 64,
        kernel_size: Sequence[int] | int = 5,
        kernel_c: int = 1,
        stride_conv: int = 1,
        pool: int = 2,
        stride_pool: int = 2,
        # Valid options : NONE, CSE, SSE, CSSE
        se_block: str = "None",
        drop_out: float = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        adn_ordering: str = "NA",
    ) -> None:
        self.act = act
        self.norm = norm
        self.adn_ordering = adn_ordering
        super().__init__()
        se_layer = self.get_selayer(num_filters, se_block)
        dropout_layer = get_dropout_layer(name=("dropout", {"p": drop_out}), dropout_dim=2)
        max_pool = get_pool_layer(
            name=("max", {"kernel_size": pool, "stride": stride_pool, "return_indices": True, "ceil_mode": True}),
            spatial_dims=2,
        )
        # for the unpooling layer there is currently no Monai implementation available, return to torch implementation
        un_pool = nn.MaxUnpool2d(kernel_size=pool, stride=stride_pool)

        # sequence of convolutional strides (like in UNet) not needed as they are always stride_conv. This defaults to 1.
        def _create_model(layer: int) -> nn.Module:
            """
            Builds the QuickNAT structure from the bottom up by recursing down to the bottelneck layer, then creating sequential
            blocks containing the decoder, a skip connection around the previous block, and the encoder.
            At the last layer a classifier block is added to the Sequential.

            Args:
                layer = inversproportional to the layers left to create
            """
            subblock: nn.Module
            if layer < 4:
                subblock = _create_model(layer + 1)

            else:
                subblock = Bottleneck(num_filters, se_layer, dropout_layer, max_pool, un_pool, kernel_size, num_filters)

            if layer == 1:
                down = ConvConcatDenseBlock(num_channels, se_layer, dropout_layer, kernel_size, num_filters)
                up = ConvConcatDenseBlock(num_filters * 2, se_layer, dropout_layer, kernel_size, num_filters)
                classifier = ClassifierBlock(2, num_filters, num_classes, stride_conv, kernel_c)
                return SequentialWithIdx(down, SkipConnectionWithIdx(subblock), up, classifier)
            else:
                up = Decoder(num_filters * 2, un_pool, se_layer, dropout_layer, kernel_size, num_filters)
                down = Encoder(num_filters, max_pool, se_layer, dropout_layer, kernel_size, num_filters)
                return SequentialWithIdx(down, SkipConnectionWithIdx(subblock), up)

        self.model = _create_model(1)

    def get_selayer(self, n_filters, se_block_type="None"):
        """
        Returns the SEBlock defined in the initialization of the QuickNAT model.

        Args:
            n_filters: encoding half of the layer
            se_block_type: defaults to None. Valid options are None, CSE, SSE, CSSE
        Returns: Appropriate SEBlock. SSE and CSSE not implemented in Monai yet.
        """
        if se_block_type == "CSE":
            return se.ChannelSELayer(2, n_filters)
        # not implemented in squeeze_and_excitation in monai use other squeeze_and_excitation import:
        elif se_block_type == "SSE" or se_block_type == "CSSE":
            # Throw error if squeeze_and_excitation is not installed
            if not flag:
                raise ImportError("Please install squeeze_and_excitation locally to use SpatialSELayer")
            if se_block_type == "SSE":
                return se1.SpatialSELayer(n_filters)
            else:
                return se1.ChannelSpatialSELayer(n_filters)
        else:
            return None

    # TODO: Do I include this:
    def enable_test_dropout(self):
        """
        Enables test time drop out for uncertainity
        :return:
        """
        attr_dict = self.__dict__["_modules"]
        for i in range(1, 5):
            encode_block, decode_block = (attr_dict["encode" + str(i)], attr_dict["decode" + str(i)])
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input, _ = self.model(input, None)
        return input


# Should go into a layers file but not clear which exact one.


class SequentialWithIdx(nn.Sequential):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor.
    Own implementation to work with the new indices in the forward pass.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, indices):
        for module in self:
            input, indices = module(input, indices)
        return input, indices
