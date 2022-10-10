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
import operator
import re
from functools import reduce
from typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.utils import model_zoo

from monai.networks.blocks import BasicEncoder
from monai.networks.layers.factories import Act, Conv, Pad, Pool
from monai.networks.layers.utils import get_norm_layer
from monai.utils.module import look_up_option

__all__ = [
    "EfficientNet",
    "EfficientNetBN",
    "get_efficientnet_image_size",
    "drop_connect",
    "EfficientNetBNFeatures",
    "BlockArgs",
    "EfficientNetEncoder",
]

efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
}

url_map = {
    "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
    "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
    "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
    "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
    "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
    "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
    "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
    "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
    # trained with adversarial examples, simplify the name to decrease string length
    "b0-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth",
    "b1-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth",
    "b2-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth",
    "b3-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth",
    "b4-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth",
    "b5-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth",
    "b6-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth",
    "b7-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth",
    "b8-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth",
}


class MBConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        image_size: List[int],
        expand_ratio: int,
        se_ratio: Optional[float],
        id_skip: Optional[bool] = True,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        drop_connect_rate: Optional[float] = 0.2,
    ) -> None:
        """
        Mobile Inverted Residual Bottleneck Block.

        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: size of the kernel for conv ops.
            stride: stride to use for conv ops.
            image_size: input image resolution.
            expand_ratio: expansion ratio for inverted bottleneck.
            se_ratio: squeeze-excitation ratio for se layers.
            id_skip: whether to use skip connection.
            norm: feature normalization type and arguments. Defaults to batch norm.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.

        References:
            [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
            [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
            [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
        """
        super().__init__()

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type = Conv["conv", spatial_dims]
        adaptivepool_type = Pool["adaptiveavg", spatial_dims]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.id_skip = id_skip
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.drop_connect_rate = drop_connect_rate

        if (se_ratio is not None) and (0.0 < se_ratio <= 1.0):
            self.has_se = True
            self.se_ratio = se_ratio
        else:
            self.has_se = False

        # Expansion phase (Inverted Bottleneck)
        inp = in_channels  # number of input channels
        oup = in_channels * expand_ratio  # number of output channels
        if self.expand_ratio != 1:
            self._expand_conv = conv_type(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._expand_conv_padding = _make_same_padder(self._expand_conv, image_size)

            self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=oup)
        else:
            # need to have the following to fix JIT error:
            #   "Module 'MBConvBlock' has no attribute '_expand_conv'"

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
        self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=oup)
        image_size = _calculate_output_image_size(image_size, self.stride)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            self._se_adaptpool = adaptivepool_type(1)
            num_squeezed_channels = max(1, int(in_channels * self.se_ratio))
            self._se_reduce = conv_type(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_reduce_padding = _make_same_padder(self._se_reduce, [1, 1])
            self._se_expand = conv_type(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
            self._se_expand_padding = _make_same_padder(self._se_expand, [1, 1])

        # Pointwise convolution phase
        final_oup = out_channels
        self._project_conv = conv_type(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._project_conv_padding = _make_same_padder(self._project_conv, image_size)
        self._bn2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=final_oup)

        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"](inplace=True)

    def forward(self, inputs: torch.Tensor):
        """MBConvBlock"s forward function.

        Args:
            inputs: Input tensor.

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
        if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
            # the combination of skip connection and drop connect brings about stochastic depth.
            if self.drop_connect_rate:
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient: bool = True) -> None:
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = Act["memswish"](inplace=True) if memory_efficient else Act["swish"](alpha=1.0)


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
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        drop_connect_rate: float = 0.2,
        depth_divisor: int = 8,
    ) -> None:
        """
        EfficientNet based on `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/pdf/1905.11946.pdf>`_.
        Adapted from `EfficientNet-PyTorch <https://github.com/lukemelas/EfficientNet-PyTorch>`_.

        Args:
            blocks_args_str: block definitions.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            width_coefficient: width multiplier coefficient (w in paper).
            depth_coefficient: depth multiplier coefficient (d in paper).
            dropout_rate: dropout rate for dropout layers.
            image_size: input image resolution.
            norm: feature normalization type and arguments.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.
            depth_divisor: depth divisor for channel rounding.

        """
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims can only be 1, 2 or 3.")

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv["conv", spatial_dims]
        adaptivepool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            "adaptiveavg", spatial_dims
        ]

        # decode blocks args into arguments for MBConvBlock
        blocks_args = [BlockArgs.from_string(s) for s in blocks_args_str]

        # checks for successful decoding of blocks_args_str
        if not isinstance(blocks_args, list):
            raise ValueError("blocks_args must be a list")

        if blocks_args == []:
            raise ValueError("block_args must be non-empty")

        self._blocks_args = blocks_args
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.drop_connect_rate = drop_connect_rate

        # expand input image dimensions to list
        current_image_size = [image_size] * spatial_dims

        # Stem
        stride = 2
        out_channels = _round_filters(32, width_coefficient, depth_divisor)  # number of output channels
        self._conv_stem = conv_type(self.in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self._conv_stem_padding = _make_same_padder(self._conv_stem, current_image_size)
        self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)
        current_image_size = _calculate_output_image_size(current_image_size, stride)

        # build MBConv blocks
        num_blocks = 0
        self._blocks = nn.Sequential()

        self.extract_stacks = []

        # update baseline blocks to input/output filters and number of repeats based on width and depth multipliers.
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=_round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=_round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=_round_repeats(block_args.num_repeat, depth_coefficient),
            )
            self._blocks_args[idx] = block_args

            # calculate the total number of blocks - needed for drop_connect estimation
            num_blocks += block_args.num_repeat

            if block_args.stride > 1:
                self.extract_stacks.append(idx)

        self.extract_stacks.append(len(self._blocks_args))

        # create and add MBConvBlocks to self._blocks
        idx = 0  # block index counter
        for stack_idx, block_args in enumerate(self._blocks_args):
            blk_drop_connect_rate = self.drop_connect_rate

            # scale drop connect_rate
            if blk_drop_connect_rate:
                blk_drop_connect_rate *= float(idx) / num_blocks

            sub_stack = nn.Sequential()
            # the first block needs to take care of stride and filter size increase.
            sub_stack.add_module(
                str(idx),
                MBConvBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_args.input_filters,
                    out_channels=block_args.output_filters,
                    kernel_size=block_args.kernel_size,
                    stride=block_args.stride,
                    image_size=current_image_size,
                    expand_ratio=block_args.expand_ratio,
                    se_ratio=block_args.se_ratio,
                    id_skip=block_args.id_skip,
                    norm=norm,
                    drop_connect_rate=blk_drop_connect_rate,
                ),
            )
            idx += 1  # increment blocks index counter

            current_image_size = _calculate_output_image_size(current_image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            # add remaining block repeated num_repeat times
            for _ in range(block_args.num_repeat - 1):
                blk_drop_connect_rate = self.drop_connect_rate

                # scale drop connect_rate
                if blk_drop_connect_rate:
                    blk_drop_connect_rate *= float(idx) / num_blocks

                # add blocks
                sub_stack.add_module(
                    str(idx),
                    MBConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_args.input_filters,
                        out_channels=block_args.output_filters,
                        kernel_size=block_args.kernel_size,
                        stride=block_args.stride,
                        image_size=current_image_size,
                        expand_ratio=block_args.expand_ratio,
                        se_ratio=block_args.se_ratio,
                        id_skip=block_args.id_skip,
                        norm=norm,
                        drop_connect_rate=blk_drop_connect_rate,
                    ),
                )
                idx += 1  # increment blocks index counter

            self._blocks.add_module(str(stack_idx), sub_stack)

        # sanity check to see if len(self._blocks) equal expected num_blocks
        if idx != num_blocks:
            raise ValueError("total number of blocks created != num_blocks")

        # Head
        head_in_channels = block_args.output_filters
        out_channels = _round_filters(1280, width_coefficient, depth_divisor)
        self._conv_head = conv_type(head_in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = _make_same_padder(self._conv_head, current_image_size)
        self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)

        # final linear layer
        self._avg_pooling = adaptivepool_type(1)
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(out_channels, self.num_classes)

        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"]()

        # initialize weights using Tensorflow's init method from official impl.
        self._initialize_weights()

    def set_swish(self, memory_efficient: bool = True) -> None:
        """
        Sets swish function as memory efficient (for training) or standard (for JIT export).

        Args:
            memory_efficient: whether to use memory-efficient version of swish.

        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)
        for sub_stack in self._blocks:
            for block in sub_stack:
                block.set_swish(memory_efficient)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

        Returns:
            a torch Tensor of classification prediction in shape ``(Batch, num_classes)``.
        """
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
            following weight init methods from
            `official Tensorflow EfficientNet implementation
            <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61>`_.
            Adapted from `EfficientNet-PyTorch's init method
            <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py>`_.
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
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        adv_prop: bool = False,
    ) -> None:
        """
        Generic wrapper around EfficientNet, used to initialize EfficientNet-B0 to EfficientNet-B7 models
        model_name is mandatory argument as there is no EfficientNetBN itself,
        it needs the N in [0, 1, 2, 3, 4, 5, 6, 7, 8] to be a model

        Args:
            model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2].
            pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2 and batch
                norm is used.
            progress: whether to show download progress for pretrained weights download.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            norm: feature normalization type and arguments.
            adv_prop: whether to use weights trained with adversarial examples.
                This argument only works when `pretrained` is `True`.

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
        # block args
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # check if model_name is valid model
        if model_name not in efficientnet_params.keys():
            raise ValueError(
                "invalid model_name {} found, must be one of {} ".format(
                    model_name, ", ".join(efficientnet_params.keys())
                )
            )

        # get network parameters
        weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

        # create model and initialize random weights
        super().__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=weight_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            image_size=image_size,
            drop_connect_rate=dropconnect_rate,
            norm=norm,
        )

        # only pretrained for when `spatial_dims` is 2
        if pretrained and (spatial_dims == 2):
            _load_state_dict(self, model_name, progress, adv_prop)


class EfficientNetBNFeatures(EfficientNet):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        progress: bool = True,
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        adv_prop: bool = False,
    ) -> None:
        """
        Initialize EfficientNet-B0 to EfficientNet-B7 models as a backbone, the backbone can
        be used as an encoder for segmentation and objection models.
        Compared with the class `EfficientNetBN`, the only different place is the forward function.

        This class refers to `PyTorch image models <https://github.com/rwightman/pytorch-image-models>`_.

        """
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # check if model_name is valid model
        if model_name not in efficientnet_params.keys():
            raise ValueError(
                "invalid model_name {} found, must be one of {} ".format(
                    model_name, ", ".join(efficientnet_params.keys())
                )
            )

        # get network parameters
        weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

        # create model and initialize random weights
        super().__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=weight_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            image_size=image_size,
            drop_connect_rate=dropconnect_rate,
            norm=norm,
        )

        # only pretrained for when `spatial_dims` is 2
        if pretrained and (spatial_dims == 2):
            _load_state_dict(self, model_name, progress, adv_prop)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input should have spatially N dimensions
            ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.

        Returns:
            a list of torch Tensors.
        """
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))

        features = []
        if 0 in self.extract_stacks:
            features.append(x)
        for i, block in enumerate(self._blocks):
            x = block(x)
            if i + 1 in self.extract_stacks:
                features.append(x)
        return features


class EfficientNetEncoder(EfficientNetBNFeatures, BasicEncoder):
    """
    Wrap the original efficientnet to an encoder for flexible-unet.
    """

    backbone_names = [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
        "efficientnet-b8",
        "efficientnet-l2",
    ]

    @classmethod
    def get_backbone_parameter(cls) -> List[Dict]:
        """
        Get the initialization parameter for efficientnet backbones.
        """
        parameter_list = []
        for backbone_name in cls.backbone_names:
            parameter_list.append(
                {
                    "model_name": backbone_name,
                    "pretrained": True,
                    "progress": True,
                    "spatial_dims": 2,
                    "in_channels": 3,
                    "num_classes": 1000,
                    "norm": ("batch", {"eps": 1e-3, "momentum": 0.01}),
                    "adv_prop": "ap" in backbone_name,
                }
            )
        return parameter_list

    @classmethod
    def get_output_feature_channel_list(cls) -> List[Tuple[int, ...]]:
        """
        Get number of efficientnet backbone output feature maps' channel.
        """
        return [
            (16, 24, 40, 112, 320),
            (16, 24, 40, 112, 320),
            (16, 24, 48, 120, 352),
            (24, 32, 48, 136, 384),
            (24, 32, 56, 160, 448),
            (24, 40, 64, 176, 512),
            (32, 40, 72, 200, 576),
            (32, 48, 80, 224, 640),
            (32, 56, 88, 248, 704),
            (72, 104, 176, 480, 1376),
        ]

    @classmethod
    def get_output_feature_number_list(cls) -> List[int]:
        """
        Get number of efficientnet backbone output feature maps.
        Since every backbone contains the same 5 output feature maps,
        the number list should be `[5] * 10`.
        """
        return [5] * 10

    @classmethod
    def get_encoder_name_string_list(cls) -> List[str]:
        """
        Get names of efficient backbone.
        """
        return cls.backbone_names


def get_efficientnet_image_size(model_name: str) -> int:
    """
    Get the input image size for a given efficientnet model.

    Args:
        model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b7].

    Returns:
        Image size for single spatial dimension as integer.

    """
    # check if model_name is valid model
    if model_name not in efficientnet_params.keys():
        raise ValueError(
            "invalid model_name {} found, must be one of {} ".format(model_name, ", ".join(efficientnet_params.keys()))
        )

    # return input image size (all dims equal so only need to return for one dim)
    _, _, res, _, _ = efficientnet_params[model_name]
    return res


def drop_connect(inputs: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    Drop connect layer that drops individual connections.
    Differs from dropout as dropconnect drops connections instead of whole neurons as in dropout.

    Based on `Deep Networks with Stochastic Depth <https://arxiv.org/pdf/1603.09382.pdf>`_.
    Adapted from `Official Tensorflow EfficientNet utils
    <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py>`_.

    This function is generalized for MONAI's N-Dimensional spatial activations
    e.g. 1D activations [B, C, H], 2D activations [B, C, H, W] and 3D activations [B, C, H, W, D]

    Args:
        inputs: input tensor with [B, C, dim_1, dim_2, ..., dim_N] where N=spatial_dims.
        p: probability to use for dropping connections.
        training: whether in training or evaluation mode.

    Returns:
        output: output tensor after applying drop connection.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in range of [0, 1], found {p}")

    # eval mode: drop_connect is switched off - so return input without modifying
    if not training:
        return inputs

    # train mode: calculate and apply drop_connect
    batch_size: int = inputs.shape[0]
    keep_prob: float = 1 - p
    num_dims: int = len(inputs.shape) - 2

    # build dimensions for random tensor, use num_dims to populate appropriate spatial dims
    random_tensor_shape: List[int] = [batch_size, 1] + [1] * num_dims

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor: torch.Tensor = torch.rand(random_tensor_shape, dtype=inputs.dtype, device=inputs.device)
    random_tensor += keep_prob

    # round to form binary tensor
    binary_tensor: torch.Tensor = torch.floor(random_tensor)

    # drop connect using binary tensor
    output: torch.Tensor = inputs / keep_prob * binary_tensor
    return output


def _load_state_dict(model: nn.Module, arch: str, progress: bool, adv_prop: bool) -> None:
    if adv_prop:
        arch = arch.split("efficientnet-")[-1] + "-ap"
    model_url = look_up_option(arch, url_map, None)
    if model_url is None:
        print(f"pretrained weights of {arch} is not provided")
    else:
        # load state dict from url
        model_url = url_map[arch]
        pretrain_state_dict = model_zoo.load_url(model_url, progress=progress)
        model_state_dict = model.state_dict()

        pattern = re.compile(r"(.+)\.\d+(\.\d+\..+)")
        for key, value in model_state_dict.items():
            pretrain_key = re.sub(pattern, r"\1\2", key)
            if pretrain_key in pretrain_state_dict and value.shape == pretrain_state_dict[pretrain_key].shape:
                model_state_dict[key] = pretrain_state_dict[pretrain_key]

        model.load_state_dict(model_state_dict)


def _get_same_padding_conv_nd(
    image_size: List[int], kernel_size: Tuple[int, ...], dilation: Tuple[int, ...], stride: Tuple[int, ...]
) -> List[int]:
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
        paddings for ConstantPadNd padder to be used on input tensor to conv op.
    """
    # get number of spatial dimensions, corresponds to kernel size length
    num_dims = len(kernel_size)

    # additional checks to populate dilation and stride (in case they are single entry tuples)
    if len(dilation) == 1:
        dilation = dilation * num_dims

    if len(stride) == 1:
        stride = stride * num_dims

    # equation to calculate (pad^+ + pad^-) size
    _pad_size: List[int] = [
        max((math.ceil(_i_s / _s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
        for _i_s, _k_s, _d, _s in zip(image_size, kernel_size, dilation, stride)
    ]
    # distribute paddings into pad^+ and pad^- following Tensorflow's same padding strategy
    _paddings: List[Tuple[int, int]] = [(_p // 2, _p - _p // 2) for _p in _pad_size]

    # unroll list of tuples to tuples, and then to list
    # reversed as nn.ConstantPadNd expects paddings starting with last dimension
    _paddings_ret: List[int] = [outer for inner in reversed(_paddings) for outer in inner]
    return _paddings_ret


def _make_same_padder(conv_op: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], image_size: List[int]):
    """
    Helper for initializing ConstantPadNd with SAME padding similar to Tensorflow.
    Uses output of _get_same_padding_conv_nd() to get the padding size.

    This function is generalized for MONAI's N-Dimensional spatial operations (e.g. Conv1D, Conv2D, Conv3D)

    Args:
        conv_op: nn.ConvNd operation to extract parameters for op from
        image_size: input image/feature spatial size

    Returns:
        If padding required then nn.ConstandNd() padder initialized to paddings otherwise nn.Identity()
    """
    # calculate padding required
    padding: List[int] = _get_same_padding_conv_nd(image_size, conv_op.kernel_size, conv_op.dilation, conv_op.stride)

    # initialize and return padder
    padder = Pad["constantpad", len(padding) // 2]
    if sum(padding) > 0:
        return padder(padding=padding, value=0.0)
    return nn.Identity()


def _round_filters(filters: int, width_coefficient: Optional[float], depth_divisor: float) -> int:
    """
    Calculate and round number of filters based on width coefficient multiplier and depth divisor.

    Args:
        filters: number of input filters.
        width_coefficient: width coefficient for model.
        depth_divisor: depth divisor to use.

    Returns:
        new_filters: new number of filters after calculation.
    """

    if not width_coefficient:
        return filters

    multiplier: float = width_coefficient
    divisor: float = depth_divisor
    filters_float: float = filters * multiplier

    # follow the formula transferred from official TensorFlow implementation
    new_filters: float = max(divisor, int(filters_float + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters_float:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats: int, depth_coefficient: Optional[float]) -> int:
    """
    Re-calculate module's repeat number of a block based on depth coefficient multiplier.

    Args:
        repeats: number of original repeats.
        depth_coefficient: depth coefficient for model.

    Returns:
        new repeat: new number of repeat after calculating.
    """
    if not depth_coefficient:
        return repeats

    # follow the formula transferred from official TensorFlow impl.
    return int(math.ceil(depth_coefficient * repeats))


def _calculate_output_image_size(input_image_size: List[int], stride: Union[int, Tuple[int]]):
    """
    Calculates the output image size when using _make_same_padder with a stride.
    Required for static padding.

    Args:
        input_image_size: input image/feature spatial size.
        stride: Conv2d operation"s stride.

    Returns:
        output_image_size: output image/feature spatial size.
    """

    # checks to extract integer stride in case tuple was received
    if isinstance(stride, tuple):
        all_strides_equal = all(stride[0] == s for s in stride)
        if not all_strides_equal:
            raise ValueError(f"unequal strides are not possible, got {stride}")

        stride = stride[0]

    # return output image size
    return [int(math.ceil(im_sz / stride)) for im_sz in input_image_size]


class BlockArgs(NamedTuple):
    """
    BlockArgs object to assist in decoding string notation
        of arguments for MBConvBlock definition.
    """

    num_repeat: int
    kernel_size: int
    stride: int
    expand_ratio: int
    input_filters: int
    output_filters: int
    id_skip: bool
    se_ratio: Optional[float] = None

    @staticmethod
    def from_string(block_string: str):
        """
        Get a BlockArgs object from a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: "r1_k3_s11_e1_i32_o16_se0.25".

        Returns:
            BlockArgs: namedtuple defined at the top of this function.
        """
        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # check stride
        stride_check = (
            ("s" in options and len(options["s"]) == 1)
            or (len(options["s"]) == 2 and options["s"][0] == options["s"][1])
            or (len(options["s"]) == 3 and options["s"][0] == options["s"][1] and options["s"][0] == options["s"][2])
        )
        if not stride_check:
            raise ValueError("invalid stride option received")

        return BlockArgs(
            num_repeat=int(options["r"]),
            kernel_size=int(options["k"]),
            stride=int(options["s"][0]),
            expand_ratio=int(options["e"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
        )

    def to_string(self):
        """
        Return a block string notation for current BlockArgs object

        Returns:
            A string notation of BlockArgs object arguments.
                Example: "r1_k3_s11_e1_i32_o16_se0.25_noskip".
        """
        string = "r{}_k{}_s{}{}_e{}_i{}_o{}_se{}".format(
            self.num_repeat,
            self.kernel_size,
            self.stride,
            self.stride,
            self.expand_ratio,
            self.input_filters,
            self.output_filters,
            self.se_ratio,
        )

        if not self.id_skip:
            string += "_noskip"
        return string
