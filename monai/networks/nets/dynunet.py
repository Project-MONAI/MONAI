# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Sequence, Union

import torch.nn as nn

from monai.networks.blocks.dynunet_block import *

__all__ = ["DynUNet", "DynUnet", "Dynunet"]


class DynUNet(nn.Module):
    """
    This reimplementation of a dynamic UNet (DynUNet) is based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

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
    is no less than 3 in order to have at least one downsample upsample blocks.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: [``"batch"``, ``"instance"``, ``"group"``]
            feature normalization type and arguments.
        deep_supervision: whether to add deep supervision head before output. Defaults to ``True``.
            If added, in training mode, the network will output not only the last feature maps
            (after being converted via output block), but also the previous feature maps that come
            from the intermediate up sample layers.
        deep_supr_num: number of feature maps that will output during deep supervision head. The
            value should be less than the number of up sample layers. Defaults to 1.
        res_block: whether to use residual connection based convolution blocks during the network.
            Defaults to ``True``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        norm_name: str = "instance",
        deep_supervision: bool = True,
        deep_supr_num: int = 1,
        res_block: bool = False,
    ):
        super(DynUNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.deep_supervision = deep_supervision
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        self.deep_supr_num = deep_supr_num
        self.apply(self.initialize_weights)
        self.check_kernel_stride()
        self.check_deep_supr_num()

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        assert len(kernels) == len(strides) and len(kernels) >= 3, error_msg

        for idx in range(len(kernels)):
            kernel, stride = kernels[idx], strides[idx]
            if not isinstance(kernel, int):
                error_msg = "length of kernel_size in block {} should be the same as spatial_dims.".format(idx)
                assert len(kernel) == self.spatial_dims, error_msg
            if not isinstance(stride, int):
                error_msg = "length of stride in block {} should be the same as spatial_dims.".format(idx)
                assert len(stride) == self.spatial_dims, error_msg

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        error_msg = "deep_supr_num should be less than the number of up sample layers."
        assert 1 <= deep_supr_num < num_up_layers, error_msg

    def forward(self, x):
        out = self.input_block(x)
        outputs = [out]

        for downsample in self.downsamples:
            out = downsample(out)
            outputs.insert(0, out)

        out = self.bottleneck(out)
        upsample_outs = []

        for upsample, skip in zip(self.upsamples, outputs):
            out = upsample(out, skip)
            upsample_outs.append(out)

        out = self.output_block(out)

        if self.training and self.deep_supervision:
            start_output_idx = len(upsample_outs) - 1 - self.deep_supr_num
            upsample_outs = upsample_outs[start_output_idx:-1][::-1]
            preds = [self.deep_supervision_heads[i](out) for i, out in enumerate(upsample_outs)]
            return [out] + preds

        return out

    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(
            self.spatial_dims,
            self.filters[idx],
            self.out_channels,
        )

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size)

    def get_module_list(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
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
                    "upsample_kernel_size": up_kernel,
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
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(len(self.upsamples) - 1)])

    @staticmethod
    def initialize_weights(module):
        name = module.__class__.__name__.lower()
        if "conv3d" in name or "conv2d" in name:
            nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif "norm" in name:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)


DynUnet = Dynunet = DynUNet
