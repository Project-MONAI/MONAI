# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math
import operator
import re
from functools import reduce
from typing import List

import torch
from torch import nn
from torch.utils import model_zoo

from monai.networks.layers.factories import Act, Conv, Norm, Pad, Pool

__all__ = ["EfficientNetBN", "get_efficientnet_image_size"]

efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5),
}


class MBConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: List[int],
        image_size: List[int],
        expand_ratio: int,
        se_ratio: float,
        id_skip: bool = True,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        drop_connect_rate: float = 0.2,
    ) -> None:
        """
        Mobile Inverted Residual Bottleneck Block.

        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_classes: number of output channels.
            kernel_size: size of the kernel for conv ops.
            stride: stride to use for conv ops.
            image_size: input image resolution.
            expand_ratio: expansion ratio for inverted bottleneck.
            se_ratio: squeeze-excitation ratio for se layers.
            id_skip: whether to use skip connection.
            batch_norm_momentum: momentum for batch norm.
            batch_norm_epsilon: epsilon for batch norm.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.

        References:
            [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
            [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
            [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
        """

        super().__init__()

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv["conv", spatial_dims]
        batchnorm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm["batch", spatial_dims]
        adaptivepool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            "adaptiveavg", spatial_dims
        ]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.id_skip = id_skip
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.drop_connect_rate = drop_connect_rate

        bn_mom = 1 - batch_norm_momentum  # pytorch"s difference from tensorflow
        bn_eps = batch_norm_epsilon

        # Expansion phase (Inverted Bottleneck)
        inp = in_channels  # number of input channels
        oup = in_channels * expand_ratio  # number of output channels
        if self.expand_ratio != 1:
            self._expand_conv = conv_type(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._expand_conv_padding = _make_same_padder(self._expand_conv, image_size)

            self._bn0 = batchnorm_type(num_features=oup, momentum=bn_mom, eps=bn_eps)
        else:
            # need to have the following to fix JIT error:
            # Module 'MBConvBlock' has no attribute '_expand_conv'

            # FIXME: find a better way to bypass JIT error
            self._expand_conv = nn.Identity()
            self._expand_conv_padding = nn.Identity()
            self._bn0 = nn.Identity()

        # Depthwise convolution phase
        self._depthwise_conv = conv_type(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False,
        )
        self._depthwise_conv_padding = _make_same_padder(self._depthwise_conv, image_size)
        self._bn1 = batchnorm_type(num_features=oup, momentum=bn_mom, eps=bn_eps)
        image_size = _calculate_output_image_size(image_size, self.stride)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            self._se_adaptpool = adaptivepool_type(1)
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self._se_reduce = conv_type(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_reduce_padding = _make_same_padder(self._se_reduce, (1, 1))
            self._se_expand = conv_type(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
            self._se_expand_padding = _make_same_padder(self._se_expand, (1, 1))

        # Pointwise convolution phase
        final_oup = out_channels
        self._project_conv = conv_type(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._project_conv_padding = _make_same_padder(self._project_conv, image_size)
        self._bn2 = batchnorm_type(num_features=final_oup, momentum=bn_mom, eps=bn_eps)
        self._swish = Act["memswish"]()

    def forward(self, inputs: torch.Tensor):
        """MBConvBlock"s forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(self._expand_conv_padding(x))
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(self._depthwise_conv_padding(x))
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = self._se_adaptpool(x)
            x_squeezed = self._se_reduce(self._se_reduce_padding(x_squeezed))
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(self._se_expand_padding(x_squeezed))
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(self._project_conv_padding(x))
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.in_channels, self.out_channels

        # stride needs to be a list
        is_stride_one = all([s == 1 for s in self.stride])

        if self.id_skip and is_stride_one and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if self.drop_connect_rate:
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient: bool = True) -> None:
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)


class EfficientNet(nn.Module):
    def __init__(
        self,
        blocks_args_str: List[str],
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        dropout_rate: float = 0.2,
        image_size: int = 224,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        drop_connect_rate: float = 0.2,
        depth_divisor=8,
    ) -> None:
        """
        EfficientNet based on `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/pdf/1905.11946.pdf>`_.
        Adapted from `EfficientNet-PyTorch
        <https://github.com/lukemelas/EfficientNet-PyTorch>`_.

        Args:
            blocks_args_str: block definitions.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            width_coefficient: width multiplier coefficient (w in paper).
            depth_coefficient: depth multiplier coefficient (d in paper).
            dropout_rate: dropout rate for dropout layers.
            image_size: input image resolution.
            batch_norm_momentum: momentum for batch norm.
            batch_norm_epsilon: epsilon for batch norm.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.
            depth_divisor: depth divisor for channel rounding.

        Examples::

            # for pretrained spatial 2D ImageNet
            >>> image_size = get_efficientnet_image_size("efficientnet-b0")
            >>> inputs = torch.rand(1, 3, image_size, image_size)
            >>> model = EfficientNetBN("efficientnet-b0", pretrained=True)
            >>> model.eval()
            >>> outputs = model(inputs)

            # create spatial 2D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=2)

            # create spatial 3D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=3)

            # create EfficientNetB7 for spatial 2D
            >>> model = EfficientNetBN("efficientnet-b7", spatial_dims=2)

        """
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise AssertionError("spatial_dims can only be 1, 2 or 3.")

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv["conv", spatial_dims]
        batchnorm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm["batch", spatial_dims]
        adaptivepool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            "adaptiveavg", spatial_dims
        ]

        # decode blocks args into arguments for MBConvBlock
        blocks_args = _decode_block_list(blocks_args_str)

        # checks for successful decoding of blocks_args_str
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"

        self._blocks_args = blocks_args
        self.num_classes = num_classes
        self.in_channels = in_channels

        # expand input image dimensions to list
        current_image_size = [image_size] * spatial_dims

        # Parameters for batch norm
        bn_mom = 1 - batch_norm_momentum  # 1 - bn_m to convert tensorflow's arg to pytorch bn compatible
        bn_eps = batch_norm_epsilon

        # Stem
        stride = [2]
        out_channels = _round_filters(32, width_coefficient, depth_divisor)  # number of output channels
        self._conv_stem = conv_type(self.in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self._conv_stem_padding = _make_same_padder(self._conv_stem, current_image_size)
        self._bn0 = batchnorm_type(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        current_image_size = _calculate_output_image_size(current_image_size, stride)

        # Build MBConv blocks
        self._blocks = nn.Sequential()
        num_blocks = 0

        # Update block input and output filters based on depth multiplier.
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=_round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=_round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=_round_repeats(block_args.num_repeat, depth_coefficient),
            )
            self._blocks_args[idx] = block_args

            # calculate the total number of blocks - needed for drop_connect estimation
            num_blocks += block_args.num_repeat

        # Create and add MBConvBlocks to self._blocks
        idx = 0  # block index counter
        for block_args in self._blocks_args:

            drop_connect_rate = drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / num_blocks  # scale drop connect_rate

            # The first block needs to take care of stride and filter size increase.
            self._blocks.add_module(
                str(idx),
                MBConvBlock(
                    spatial_dims,
                    block_args.input_filters,
                    block_args.output_filters,
                    block_args.kernel_size,
                    block_args.stride,
                    current_image_size,
                    block_args.expand_ratio,
                    block_args.se_ratio,
                    block_args.id_skip,
                    batch_norm_momentum,
                    batch_norm_epsilon,
                    drop_connect_rate=drop_connect_rate,
                ),
            )
            idx += 1  # increment blocks index counter

            current_image_size = _calculate_output_image_size(current_image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=[1])

            # Repeat block for num_repeat required
            for _ in range(block_args.num_repeat - 1):
                drop_connect_rate = drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / num_blocks  # scale drop connect_rate
                self._blocks.add_module(
                    str(idx),
                    MBConvBlock(
                        spatial_dims,
                        block_args.input_filters,
                        block_args.output_filters,
                        block_args.kernel_size,
                        block_args.stride,
                        current_image_size,
                        block_args.expand_ratio,
                        block_args.se_ratio,
                        block_args.id_skip,
                        batch_norm_momentum,
                        batch_norm_epsilon,
                        drop_connect_rate=drop_connect_rate,
                    ),
                )
                idx += 1  # increment blocks index counter

        # Sanity check to see if len(self._blocks) equal expected num_blocks
        assert len(self._blocks) == num_blocks

        # Head
        head_in_channels = block_args.output_filters
        out_channels = _round_filters(1280, width_coefficient, depth_divisor)
        self._conv_head = conv_type(head_in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = _make_same_padder(self._conv_head, current_image_size)
        self._bn1 = batchnorm_type(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = adaptivepool_type(1)
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(out_channels, self.num_classes)

        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"]()

        # initialize weights
        self._initialize_weights()

    def set_swish(self, memory_efficient: bool = True) -> None:
        """
        Sets swish function as memory efficient (for training) or standard (for JIT export).

        Args:
            memory_efficient: whether to use memory-efficient version of swish.

        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

        Returns:
            A torch Tensor of classification prediction in shape
            ``(Batch, num_classes)``.
        """
        # Convolution layers
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        # Pooling and final linear layer
        x = self._avg_pooling(x)

        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    def _initialize_weights(self) -> None:
        """
        Args:
            None, initializes weights for conv/linear/batchnorm layers
            following weight init methods from `official Tensorflow EfficientNet implementation <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61>`_.
            Adapted from `EfficientNet-PyTorch's init method <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py>`_.
        """
        for _, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_out = reduce(operator.mul, m.kernel_size, 1) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                fan_in = 0
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


class EfficientNetBN(EfficientNet):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        progress: bool = True,
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
    ) -> None:
        """
        Generic wrapper around EfficientNet, used to initialize EfficientNet-B0 to EfficientNet-B7 models
        model_name is mandatory argument as there is no EfficientNetBN itself, it needs the N \in [0, 1, 2, 3, 4, 5, 6, 7] to be a model

        Args:
            model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b7].
            pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2.
            progress: whether to show download progress for pretrained weights download.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.

        """
        # block args for EfficientNet-B0 to EfficientNet-B7
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]
        assert model_name in efficientnet_params.keys(), "model_name should be one of {} ".format(
            ", ".join(efficientnet_params.keys())
        )

        # get network parameters
        wc, dc, isize, dr = efficientnet_params[model_name]

        # create model and initialize random weights
        model = super(EfficientNetBN, self).__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=wc,
            depth_coefficient=dc,
            dropout_rate=dr,
            image_size=isize,
        )

        # attempt to load pretrained
        is_default_model = (spatial_dims == 2) and (in_channels == 3)
        loadable_from_file = pretrained and is_default_model

        if loadable_from_file:
            # skip loading fc layers for transfer learning applications
            load_fc = num_classes == 1000

            # only pretrained for when `spatial_dims` is 2
            _load_state_dict(self, model_name, progress, load_fc)
        else:
            print(
                "Skipping loading pretrained weights for non-default {}, pretrained={}, is_default_model={}".format(
                    model_name, pretrained, is_default_model
                )
            )


def _load_state_dict(model: nn.Module, model_name: str, progress: bool, load_fc: bool) -> None:
    url_map = {
        "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
        "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
        "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
        "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
    }
    model_url = url_map[model_name]
    state_dict = model_zoo.load_url(model_url, progress=progress)

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)
    else:
        state_dict.pop("_fc.weight")
        state_dict.pop("_fc.bias")
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == {
            "_fc.weight",
            "_fc.bias",
        }, "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)

    assert not ret.unexpected_keys, "Missing keys when loading pretrained weights: {}".format(ret.unexpected_keys)


def get_efficientnet_image_size(model_name: str) -> int:
    """
    Get the input image size for a given efficientnet model.

    Args:
        model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b7].

    Returns:
        Image size for single spatial dimension as integer.

    """
    assert model_name in efficientnet_params.keys(), "model_name should be one of {} ".format(
        ", ".join(efficientnet_params.keys())
    )
    _, _, res, _ = efficientnet_params[model_name]
    return res


def _round_filters(filters, width_coefficient=None, depth_divisor=None):
    """
    Calculate and round number of filters based on width coefficient multiplier and depth divisor.

    Args:
        filters: number of input filters.
        width_coefficient: width coefficient for model.
        depth_divisor: depth divisor to use.

    Returns:
        new_filters: new number of filters after calculation.
    """
    multiplier = width_coefficient
    if not multiplier:
        return filters
    divisor = depth_divisor
    filters *= multiplier

    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats, depth_coefficient=None):
    """
    Re-calculate module's repeat number of a block based on depth coefficient multiplier.

    Args:
        repeats: number of original repeats.
        depth_coefficient: depth coefficient for model.

    Returns:
        new repeat: new number of repeat after calculating.
    """
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    Drop connect layer that drops individual connections.
    Differs from dropout as dropconnect drops connections instead of whole neurons as in dropout.
    Based on `Deep Networks with Stochastic Depth <https://arxiv.org/pdf/1603.09382.pdf>`_.
    Adapted from `Official Tensorflow EfficientNet utils <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py>`_.

    Args:
        input: input tensor with [B, C, dim_1, dim_2, ..., dim_N] where N=spatial_dims.
        p: probability to use for dropping connections.
        training: whether in training or evaluation mode.

    Returns:
        output: output tensor after applying drop connection.
    """
    assert 0 <= p <= 1, "p must be in range of [0,1]"

    if not training:
        return inputs

    batch_size: int = inputs.shape[0]
    keep_prob: float = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    # random_tensor = keep_prob
    random_tensor: torch.Tensor = torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    random_tensor += keep_prob

    binary_tensor: torch.Tensor = torch.floor(random_tensor)

    output: torch.Tensor = inputs / keep_prob * binary_tensor
    return output


def _calculate_output_image_size(input_image_size, stride):
    """
    Calculates the output image size when using _make_same_padder with a stride.
    Necessary for static padding.

    Args:
        input_image_size: input image/feature spatial size.
        stride: Conv2d operation"s stride.

    Returns:
        output_image_size: output image/feature spatial size.
    """
    if input_image_size is None:
        return None

    num_dims = len(input_image_size)
    assert isinstance(stride, list)

    if len(stride) != len(input_image_size):
        stride = stride * num_dims

    return [int(math.ceil(im_sz / st)) for im_sz, st in zip(input_image_size, stride)]


def _get_same_padding_convNd(image_size, kernel_size, dilation, stride):
    """
    Helper for getting padding (nn.ConstantPadNd) to be used to get SAME padding
    conv operations similar to Tensorflow's SAME padding.

    This function is generalized for MONAI's N-Dimensional spatial operations (e.g. Conv1D, Conv2D, Conv3D)

    Args:
        image_size: input image/feature spatial size.
        kernel_size: conv kernel's spatial size.
        dilation: conv dilation rate for Atrous conv.
        stride: stride for conv operation.

    Returns:
        paddings for ConstantPadXd padder to be used on input tensor to conv op.
    """
    num_dims = len(kernel_size)

    # additional checks to populate dilation and stride (in case they are single entry list)
    if len(dilation) == 1:
        dilation = dilation * num_dims

    if len(stride) == 1:
        stride = stride * num_dims

    # equation to calculate (pad^+ + pad^-) size
    _pad_size = [
        max((math.ceil(_i_s / _s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
        for _i_s, _k_s, _d, _s in zip(image_size, kernel_size, dilation, stride)
    ]
    # distribute paddings into pad^+ and pad^- following Tensorflow's same padding strategy
    _paddings = [(_p // 2, _p - _p // 2) for _p in _pad_size]

    # unroll list of tuples to tuples,
    # reversed as nn.ConstantPadXd expects paddings starting with last dimenion
    _paddings = [outer for inner in reversed(_paddings) for outer in inner]
    return _paddings


def _make_same_padder(conv_op, image_size):
    """
    Helper for initializing ConstantPadNd with SAME padding similar to Tensorflow.
    Uses output of _get_same_padding_convNd() to get the padding size.
    Generalized for N-Dimensional spatial operatoins e.g. Conv1D, Conv2D, Conv3D

    Args:
        conv_op: nn.ConvNd operation to extract parameters for op from
        image_size: input image/feature spatial size

    Returns:
        If padding required then nn.ConstandNd() padder initialized to paddings otherwise nn.Identity()
    """
    # calculate padding required
    padding = _get_same_padding_convNd(image_size, conv_op.kernel_size, conv_op.dilation, conv_op.stride)

    # initialize and return padder
    padder = Pad["constantpad", len(padding) // 2]
    if sum(padding) > 0:
        return padder(padding=padding, value=0.0)
    else:
        return nn.Identity()


def _decode_block_list(string_list):
    """
    Decode a list of string notations to specify blocks inside the network.

    Args:
        string_list: a list of strings, each string is a notation of block.

    Returns:
        blocks_args: a list of BlockArgs namedtuples of block args.
    """
    # Parameters for an individual model block
    BlockArgs = collections.namedtuple(
        "BlockArgs",
        [
            "num_repeat",
            "kernel_size",
            "stride",
            "expand_ratio",
            "input_filters",
            "output_filters",
            "se_ratio",
            "id_skip",
        ],
    )
    BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

    def _decode_block_string(block_string):
        """
        Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: "r1_k3_s11_e1_i32_o16_se0.25".

        Returns:
            BlockArgs: namedtuple defined at the top of this function.
        """
        assert isinstance(block_string, str)

        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert ("s" in options and len(options["s"]) == 1) or (
            len(options["s"]) == 2 and options["s"][0] == options["s"][1]
        )

        return BlockArgs(
            num_repeat=int(options["r"]),
            kernel_size=int(options["k"]),
            stride=[int(options["s"][0])],
            expand_ratio=int(options["e"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            se_ratio=float(options["se"]) if "se" in options else None,
            id_skip=("noskip" not in block_string),
        )

    assert isinstance(string_list, list)
    blocks_args = []
    for b_s in string_list:
        blocks_args.append(_decode_block_string(b_s))
    return blocks_args
