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

"""Implementation based on: https://github.com/lukemelas/EfficientNet-PyTorch
#With significant modifications to refactor/rewrite the code for MONAI
"""
import collections
import math
import re
from typing import List

import torch
from torch import nn
from torch.utils import model_zoo

from monai.networks.layers.factories import Act, Conv, Norm, Pad, Pool

__all__ = ["EfficientNetBN", "get_efficientnet_image_size"]

efficientnet_params = {
    # Coefficients:   width,depth,res,dropout
    "efficientnet-b0": (1.0, 1.0, 224, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5),
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
}


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        image_size: List[int],
        expand_ratio: float = 1.0,
        se_ratio: float = 0.25,
        id_skip: bool = True,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.id_skip = id_skip
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self._drop_connect_rate = drop_connect_rate

        nd_conv = Conv["conv", self.spatial_dims]
        nd_batchnorm = Norm["batch", self.spatial_dims]
        nd_adaptivepool = Pool["adaptiveavg", self.spatial_dims]

        bn_mom = 1 - batch_norm_momentum  # pytorch"s difference from tensorflow
        bn_eps = batch_norm_epsilon

        # Expansion phase (Inverted Bottleneck)
        inp = in_channels  # number of input channels
        oup = in_channels * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nd_conv(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._expand_conv_padding = _make_same_padder(self._expand_conv, image_size)

            self._bn0 = nd_batchnorm(num_features=oup, momentum=bn_mom, eps=bn_eps)

        # Depthwise convolution phase
        self._depthwise_conv = nd_conv(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False,
        )
        self._depthwise_conv_padding = _make_same_padder(self._depthwise_conv, image_size)
        self._bn1 = nd_batchnorm(num_features=oup, momentum=bn_mom, eps=bn_eps)
        image_size = _calculate_output_image_size(image_size, self.stride)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            self._se_adaptpool = nd_adaptivepool(1)
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self._se_reduce = nd_conv(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_reduce_padding = _make_same_padder(self._se_reduce, (1, 1))
            self._se_expand = nd_conv(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
            self._se_expand_padding = _make_same_padder(self._se_expand, (1, 1))

        # Pointwise convolution phase
        final_oup = out_channels
        self._project_conv = nd_conv(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._project_conv_padding = _make_same_padder(self._project_conv, image_size)
        self._bn2 = nd_batchnorm(num_features=final_oup, momentum=bn_mom, eps=bn_eps)
        # self._swish = MemoryEfficientSwish()
        self._swish = Act["memswish"]()

    def forward(self, inputs):
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
            x = self._expand_conv(self._expand_conv_padding(inputs))
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
        assert isinstance(self.stride, list)
        is_stride_one = all([s == 1 for s in self.stride])

        if self.id_skip and is_stride_one and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if self._drop_connect_rate:
                x = drop_connect(x, p=self._drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:


        import torch
        >>> from monai.networks.nets import get_efficientnet_image_size, EfficientNetBN
        >>> image_size = get_efficientnet_image_size("efficientnet-b0")
        >>> inputs = torch.rand(1, 3, image_size, image_size)
        >>> model = EfficientNetBN("efficientnet-b0")
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(
        self,
        blocks_args: List[str],
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
    ):
        super().__init__()

        blocks_args = _decode_block_list(blocks_args)

        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"

        self._blocks_args = blocks_args

        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Batch norm parameters
        bn_mom = 1 - batch_norm_momentum
        bn_eps = batch_norm_epsilon

        if isinstance(image_size, int):
            image_size = [image_size] * self.spatial_dims

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        nd_conv = Conv["conv", self.spatial_dims]
        nd_batchnorm = Norm["batch", self.spatial_dims]
        nd_adaptivepool = Pool["adaptiveavg", self.spatial_dims]

        # Stem
        stride = [2]
        out_channels = _round_filters(32, width_coefficient, depth_divisor)  # number of output channels
        self._conv_stem = nd_conv(self.in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self._conv_stem_padding = _make_same_padder(self._conv_stem, image_size)
        self._bn0 = nd_batchnorm(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = _calculate_output_image_size(image_size, stride)

        # Build blocks
        self._blocks = []
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

        idx = 0
        for block_args in self._blocks_args:

            drop_connect_rate = drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / num_blocks  # scale drop connect_rate

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(
                    self.spatial_dims,
                    block_args.input_filters,
                    block_args.output_filters,
                    block_args.kernel_size,
                    block_args.stride,
                    image_size,
                    block_args.expand_ratio,
                    block_args.se_ratio,
                    block_args.id_skip,
                    batch_norm_momentum,
                    batch_norm_epsilon,
                    drop_connect_rate=drop_connect_rate,
                )
            )
            idx += 1

            image_size = _calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=[1])
            for _ in range(block_args.num_repeat - 1):
                drop_connect_rate = drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / num_blocks  # scale drop connect_rate
                self._blocks.append(
                    MBConvBlock(
                        self.spatial_dims,
                        block_args.input_filters,
                        block_args.output_filters,
                        block_args.kernel_size,
                        block_args.stride,
                        image_size,
                        block_args.expand_ratio,
                        block_args.se_ratio,
                        block_args.id_skip,
                        batch_norm_momentum,
                        batch_norm_epsilon,
                        drop_connect_rate=drop_connect_rate,
                    )
                )
                idx += 1
        self._blocks = nn.Sequential(*self._blocks)
        assert len(self._blocks) == num_blocks

        # Head
        head_in_channels = block_args.output_filters
        out_channels = _round_filters(1280, width_coefficient, depth_divisor)
        self._conv_head = nd_conv(head_in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = _make_same_padder(self._conv_head, image_size)
        self._bn1 = nd_batchnorm(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nd_adaptivepool(1)
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(out_channels, self.num_classes)

        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using set_swish function call
        self._swish = Act["memswish"]()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward(self, inputs):
        """EfficientNet"s forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
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

    def _initialize_weight(self):
        # weight init as per Tensorflow Official impl
        # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61
        # code based on: https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py
        for _, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_out = math.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)  # fan-out
                fan_in = 0
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


class EfficientNetBN(EfficientNet):
    # model_name mandatory as there is is EfficientNetBN itself, it needs the N \in [0, 1, 2, 3, 4, 5, 6, 7, 8] to be a model
    def __init__(self, model_name, pretrained=True, progress=True, spatial_dims=2, in_channels=3, num_classes=1000):
        block_args = [
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

        wc, dc, isize, dr = efficientnet_params[model_name]
        model = super(EfficientNetBN, self).__init__(
            block_args,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=wc,
            depth_coefficient=dc,
            dropout_rate=dr,
            image_size=isize,
        )

        is_default_model = (spatial_dims == 2) and (in_channels == 3)
        pretrained = pretrained and model_name in url_map

        loadable_from_file = pretrained and is_default_model

        if loadable_from_file:
            # skip loading fc layers for transfer learning applications
            load_fc = num_classes == 1000
            model_url = url_map[model_name]

            # only pretrained for when `spatial_dims` is 2
            _load_state_dict(self, model_url, progress, load_fc)
        else:
            print(
                "Skipping loading pretrained weights for non-default {}, pretrained={}, is_default_model={}".format(
                    model_name, pretrained, is_default_model
                )
            )
            print("Initializing weights for {}".format(model_name))
            self._initialize_weight()


def _load_state_dict(model: nn.Module, model_url: str, progress: bool, load_fc: bool) -> bool:
    state_dict = model_zoo.load_url(model_url, progress=progress)
    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)
    else:
        state_dict.pop("_fc.weight")
        state_dict.pop("_fc.bias")
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(
            "_fc.weight", "_fc.bias"
        ), "Missing keys when loading pretrained weights: {}".format(ret.missing_keys)

    assert not ret.unexpected_keys, "Missing keys when loading pretrained weights: {}".format(ret.unexpected_keys)


def get_efficientnet_image_size(model_name):
    """Get the input image size for a given efficientnet model."""
    assert model_name in efficientnet_params.keys(), "model_name should be one of {} ".format(
        ", ".join(efficientnet_params.keys())
    )
    _, _, res, _ = efficientnet_params[model_name]
    return res


def _round_filters(filters, width_coefficient=None, depth_divisor=None):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
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
    """Calculate module"s repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, "p must be in range of [0,1]"

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def _calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation"s stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None

    num_dims = len(input_image_size)
    assert isinstance(stride, list)

    if len(stride) != len(input_image_size):
        stride = stride * num_dims

    return [int(math.ceil(im_sz / st)) for im_sz, st in zip(input_image_size, stride)]


def _get_same_padding_conv2d(image_size, kernel_size, dilation, stride):
    num_dims = len(kernel_size)

    # additional checks to populate dilation and stride (in case they are integers or single entry list)
    if isinstance(stride, int):
        stride = (stride,) * num_dims
    elif len(stride) == 1:
        stride = stride * num_dims

    if isinstance(dilation, int):
        dilation = (dilation,) * num_dims
    elif len(dilation) == 1:
        dilation = dilation * num_dims

    # _kernel_size = kernel_size
    _pad_size = [
        max((math.ceil(_i_s / _s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
        for _i_s, _k_s, _d, _s in zip(image_size, kernel_size, dilation, stride)
    ]
    _paddings = [(_p // 2, _p - _p // 2) for _p in _pad_size]

    # unroll list of tuples to tuples,
    # reversed as constandpadnd expects paddings starting with last dimenion
    _paddings = [outer for inner in reversed(_paddings) for outer in inner]
    return _paddings


def _make_same_padder(conv_op, image_size):
    padding = _get_same_padding_conv2d(image_size, conv_op.kernel_size, conv_op.dilation, conv_op.stride)
    padder = Pad["constantpad", len(padding) // 2]
    if sum(padding) > 0:
        return padder(padding=padding, value=0)
    else:
        return nn.Identity()


def _decode_block_list(string_list):
    """Decode a list of string notations to specify blocks inside the network.

    Args:
        string_list (list[str]): A list of strings, each string is a notation of block.

    Returns:
        blocks_args: A list of BlockArgs namedtuples of block args.
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
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: "r1_k3_s11_e1_i32_o16_se0.25_noskip".

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
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
