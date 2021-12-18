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


from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock

__all__ = ["DynUNet", "DynUnet", "Dynunet"]


class DynUNetSkipLayer(nn.Module):
    """
    Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip connection.
    The member `next_layer` may refer to instances of this class or the final bottleneck layer at the bottom the UNet
    structure. The purpose of using a recursive class like this is to get around the Torchscript restrictions on
    looping over lists of layers and accumulating lists of output tensors which must be indexed. The `heads` list is
    shared amongst all the instances of this class and is used to store the output from the supervision heads during
    forward passes of the network.
    """

    heads: Optional[List[torch.Tensor]]

    def __init__(self, index, downsample, upsample, next_layer, heads=None, super_head=None):
        super().__init__()
        self.downsample = downsample
        self.next_layer = next_layer
        self.upsample = upsample
        self.super_head = super_head
        self.heads = heads
        self.index = index

    def forward(self, x):
        downout = self.downsample(x)
        nextout = self.next_layer(downout)
        upout = self.upsample(nextout, downout)
        if self.super_head is not None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout)

        return upout


class DynUNet(nn.Module):
    """
    This reimplementation of a dynamic UNet (DynUNet) is based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    `Optimized U-Net for Brain Tumor Segmentation <https://arxiv.org/pdf/2110.03352.pdf>`_.

    This model is more flexible compared with ``monai.networks.nets.UNet`` in three
    places:

        - Residual connection is supported in conv blocks.
        - Anisotropic kernel sizes and strides can be used in each layers.
        - Deep supervision heads can be added.

    The model supports 2D or 3D inputs and is consisted with four kinds of blocks:
    one input block, `n` downsample blocks, one bottleneck and `n+1` upsample blocks. Where, `n>0`.
    The first and last kernel and stride values of the input sequences are used for input block and
    bottleneck respectively, and the rest value(s) are used for downsample and upsample blocks.
    Therefore, pleasure ensure that the length of input sequences (``kernel_size`` and ``strides``)
    is no less than 3 in order to have at least one downsample and upsample blocks.

    To meet the requirements of the structure, the input size for each spatial dimension should be divisible
    by `2 * the product of all strides in the corresponding dimension`. The output size for each spatial dimension
    equals to the input size of the corresponding dimension divided by the stride in strides[0].
    For example, if `strides=((1, 2, 4), 2, 1, 1)`, the minimal spatial size of the input is `(8, 16, 32)`, and
    the spatial size of the output is `(8, 8, 8)`.

    For backwards compatibility with old weights, please set `strict=False` when calling `load_state_dict`.

    Usage example with medical segmentation decathlon dataset is available at:
    https://github.com/Project-MONAI/tutorials/tree/master/modules/dynunet_pipeline.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        upsample_kernel_size: convolution kernel size for transposed convolution layers. The values should
            equal to strides[1:].
        filters: number of output channels for each blocks. Different from nnU-Net, in this implementation we add
            this argument to make the network more flexible. As shown in the third reference, one way to determine
            this argument is like:
            ``[64, 96, 128, 192, 256, 384, 512, 768, 1024][: len(strides)]``.
            The above way is used in the network that wins task 1 in the BraTS21 Challenge.
            If not specified, the way which nnUNet used will be employed. Defaults to ``None``.
        dropout: dropout ratio. Defaults to no dropout.
        norm_name: feature normalization type and arguments. Defaults to ``INSTANCE``.
        act_name: activation layer type and arguments. Defaults to ``leakyrelu``.
        deep_supervision: whether to add deep supervision head before output. Defaults to ``False``.
            If ``True``, in training mode, the forward function will output not only the final feature map
            (from `output_block`), but also the feature maps that come from the intermediate up sample layers.
            In order to unify the return type (the restriction of TorchScript), all intermediate
            feature maps are interpolated into the same size as the final feature map and stacked together
            (with a new dimension in the first axis)into one single tensor.
            For instance, if there are two intermediate feature maps with shapes: (1, 2, 16, 12) and
            (1, 2, 8, 6), and the final feature map has the shape (1, 2, 32, 24), then all intermediate feature maps
            will be interpolated into (1, 2, 32, 24), and the stacked tensor will has the shape (1, 3, 2, 32, 24).
            When calculating the loss, you can use torch.unbind to get all feature maps can compute the loss
            one by one with the ground truth, then do a weighted average for all losses to achieve the final loss.
        deep_supr_num: number of feature maps that will output during deep supervision head. The
            value should be larger than 0 and less than the number of up sample layers.
            Defaults to 1.
        res_block: whether to use residual connection based convolution blocks during the network.
            Defaults to ``False``.
        trans_bias: whether to set the bias parameter in transposed convolution layers. Defaults to ``False``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.trans_bias = trans_bias
        if filters is not None:
            self.filters = filters
            self.check_filters()
        else:
            self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num
        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * self.deep_supr_num
        if self.deep_supervision:
            self.deep_supervision_heads = self.get_deep_supervision_heads()
            self.check_deep_supr_num()

        self.apply(self.initialize_weights)
        self.check_kernel_stride()

        def create_skips(index, downsamples, upsamples, bottleneck, superheads=None):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """

            if len(downsamples) != len(upsamples):
                raise ValueError(f"{len(downsamples)} != {len(upsamples)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck

            if superheads is None:
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck)
                return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)

            super_head_flag = False
            if index == 0:  # don't associate a supervision head with self.input_block
                rest_heads = superheads
            else:
                if len(superheads) > 0:
                    super_head_flag = True
                    rest_heads = superheads[1:]
                else:
                    rest_heads = nn.ModuleList()

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, superheads=rest_heads)
            if super_head_flag:
                return DynUNetSkipLayer(
                    index,
                    downsample=downsamples[0],
                    upsample=upsamples[0],
                    next_layer=next_layer,
                    heads=self.heads,
                    super_head=superheads[0],
                )

            return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)

        if not self.deep_supervision:
            self.skip_layers = create_skips(
                0, [self.input_block] + list(self.downsamples), self.upsamples[::-1], self.bottleneck
            )
        else:
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1],
                self.bottleneck,
                superheads=self.deep_supervision_heads,
            )

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if len(kernels) != len(strides) or len(kernels) < 3:
            raise ValueError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = f"length of kernel_size in block {idx} should be the same as spatial_dims."
                if len(kernel) != self.spatial_dims:
                    raise ValueError(error_msg)
            if not isinstance(stride, int):
                error_msg = f"length of stride in block {idx} should be the same as spatial_dims."
                if len(stride) != self.spatial_dims:
                    raise ValueError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise ValueError("deep_supr_num should be less than the number of up sample layers.")
        if deep_supr_num < 1:
            raise ValueError("deep_supr_num should be larger than 0.")

    def check_filters(self):
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        else:
            self.filters = filters[: len(self.strides)]

    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(interpolate(feature_map, out.shape[2:]))
            return torch.stack(out_all, dim=1)
        return out

    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=self.dropout)

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size, trans_bias=self.trans_bias
        )

    def get_module_list(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(self.deep_supr_num)])

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


DynUnet = Dynunet = DynUNet
