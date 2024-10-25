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

from collections import OrderedDict
from collections.abc import Callable, Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from monai.networks.blocks import ADN
from monai.networks.blocks.aspp import SimpleASPP
from monai.networks.blocks.backbone_fpn_utils import BackboneWithFPN
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork
from monai.networks.layers.factories import Conv, Norm
from monai.networks.layers.utils import get_norm_layer
from monai.networks.nets.resnet import ResNet, ResNetBottleneck

__all__ = [
    "AttentionModule",
    "Daf3dASPP",
    "Daf3dResNetBottleneck",
    "Daf3dResNetDilatedBottleneck",
    "Daf3dResNet",
    "Daf3dBackbone",
    "Daf3dFPN",
    "Daf3dBackboneWithFPN",
    "DAF3D",
]


class AttentionModule(nn.Module):
    """
    Attention Module as described in 'Deep Attentive Features for Prostate Segmentation in 3D Transrectal Ultrasound'
    <https://arxiv.org/pdf/1907.01743.pdf>. Returns refined single layer feature (SLF) and attentive map

    Args:
        spatial_dims: dimension of inputs.
        in_channels: number of input channels (channels of slf and mlf).
        out_channels: number of output channels (channels of attentive map and refined slf).
        norm: normalization type.
        act: activation type.
    """

    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        norm=("group", {"num_groups": 32, "num_channels": 64}),
        act="PRELU",
    ):
        super().__init__()

        self.attentive_map = nn.Sequential(
            Convolution(spatial_dims, in_channels, out_channels, kernel_size=1, norm=norm, act=act),
            Convolution(spatial_dims, out_channels, out_channels, kernel_size=3, padding=1, norm=norm, act=act),
            Convolution(
                spatial_dims, out_channels, out_channels, kernel_size=3, padding=1, adn_ordering="A", act="SIGMOID"
            ),
        )
        self.refine = nn.Sequential(
            Convolution(spatial_dims, in_channels, out_channels, kernel_size=1, norm=norm, act=act),
            Convolution(spatial_dims, out_channels, out_channels, kernel_size=3, padding=1, norm=norm, act=act),
            Convolution(spatial_dims, out_channels, out_channels, kernel_size=3, padding=1, norm=norm, act=act),
        )

    def forward(self, slf, mlf):
        att = self.attentive_map(torch.cat((slf, mlf), 1))
        out = self.refine(torch.cat((slf, att * mlf), 1))
        return (out, att)


class Daf3dASPP(SimpleASPP):
    """
    Atrous Spatial Pyramid Pooling module as used in 'Deep Attentive Features for Prostate Segmentation in
    3D Transrectal Ultrasound' <https://arxiv.org/pdf/1907.01743.pdf>. Core functionality as in SimpleASPP, but after each
    layerwise convolution a group normalization is added. Further weight initialization for convolutions is provided in
    _init_weight(). Additional possibility to specify the number of final output channels.

    Args:
        spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
        in_channels: number of input channels.
        conv_out_channels: number of output channels of each atrous conv.
        out_channels: number of output channels of final convolution.
            If None, uses len(kernel_sizes) * conv_out_channels
        kernel_sizes: a sequence of four convolutional kernel sizes.
            Defaults to (1, 3, 3, 3) for four (dilated) convolutions.
        dilations: a sequence of four convolutional dilation parameters.
            Defaults to (1, 2, 4, 6) for four (dilated) convolutions.
        norm_type: final kernel-size-one convolution normalization type.
            Defaults to batch norm.
        acti_type: final kernel-size-one convolution activation type.
            Defaults to leaky ReLU.
        bias: whether to have a bias term in convolution blocks. Defaults to False.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.

    Raises:
        ValueError: When ``kernel_sizes`` length differs from ``dilations``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        conv_out_channels: int,
        out_channels: int | None = None,
        kernel_sizes: Sequence[int] = (1, 3, 3, 3),
        dilations: Sequence[int] = (1, 2, 4, 6),
        norm_type: tuple | str | None = "BATCH",
        acti_type: tuple | str | None = "LEAKYRELU",
        bias: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims, in_channels, conv_out_channels, kernel_sizes, dilations, norm_type, acti_type, bias
        )

        # add normalization after each atrous convolution, initializes weights
        new_convs = nn.ModuleList()
        for _conv in self.convs:
            tmp_conv = Convolution(1, 1, 1)
            tmp_conv.conv = _conv
            tmp_conv.adn = ADN(ordering="N", norm=norm_type, norm_dim=1)
            tmp_conv = self._init_weight(tmp_conv)
            new_convs.append(tmp_conv)
        self.convs = new_convs

        # change final convolution to different out_channels
        if out_channels is None:
            out_channels = len(kernel_sizes) * conv_out_channels

        self.conv_k1 = Convolution(
            spatial_dims=3,
            in_channels=len(kernel_sizes) * conv_out_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm_type,
            act=acti_type,
        )

    def _init_weight(self, conv):
        for m in conv.modules():
            if isinstance(m, nn.Conv3d):  # true for conv.conv
                torch.nn.init.kaiming_normal_(m.weight)
        return conv


class Daf3dResNetBottleneck(ResNetBottleneck):
    """
    ResNetBottleneck block as used in 'Deep Attentive Features for Prostate Segmentation in 3D
    Transrectal Ultrasound' <https://arxiv.org/pdf/1907.01743.pdf>.
    Instead of Batch Norm Group Norm is used, instead of ReLU PReLU activation is used.
    Initial expansion is 2 instead of 4 and second convolution uses groups.

    Args:
        in_planes: number of input channels.
        planes: number of output channels (taking expansion into account).
        spatial_dims: number of spatial dimensions of the input image.
        stride: stride to use for second conv layer.
        downsample: which downsample layer to use.
        norm: which normalization layer to use. Defaults to group.
    """

    expansion = 2

    def __init__(
        self, in_planes, planes, spatial_dims=3, stride=1, downsample=None, norm=("group", {"num_groups": 32})
    ):
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        norm_layer = partial(get_norm_layer, name=norm, spatial_dims=spatial_dims)

        # in case downsample uses batch norm, change to group norm
        if isinstance(downsample, nn.Sequential):
            downsample = nn.Sequential(
                conv_type(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channels=planes * self.expansion),
            )

        super().__init__(in_planes, planes, spatial_dims, stride, downsample)

        # change norm from batch to group norm
        self.bn1 = norm_layer(channels=planes)
        self.bn2 = norm_layer(channels=planes)
        self.bn3 = norm_layer(channels=planes * self.expansion)

        # adapt second convolution to work with groups
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, stride=stride, groups=32, bias=False)

        # adapt activation function
        self.relu = nn.PReLU()


class Daf3dResNetDilatedBottleneck(Daf3dResNetBottleneck):
    """
    ResNetDilatedBottleneck as used in 'Deep Attentive Features for Prostate Segmentation in 3D
    Transrectal Ultrasound' <https://arxiv.org/pdf/1907.01743.pdf>.
    Same as Daf3dResNetBottleneck but dilation of 2 is used in second convolution.
    Args:
        in_planes: number of input channels.
        planes: number of output channels (taking expansion into account).
        spatial_dims: number of spatial dimensions of the input image.
        stride: stride to use for second conv layer.
        downsample: which downsample layer to use.
    """

    def __init__(
        self, in_planes, planes, spatial_dims=3, stride=1, downsample=None, norm=("group", {"num_groups": 32})
    ):
        super().__init__(in_planes, planes, spatial_dims, stride, downsample, norm)

        # add dilation in second convolution
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        self.conv2 = conv_type(
            planes, planes, kernel_size=3, stride=stride, padding=2, dilation=2, groups=32, bias=False
        )


class Daf3dResNet(ResNet):
    """
    ResNet as used in 'Deep Attentive Features for Prostate Segmentation in 3D Transrectal Ultrasound'
    <https://arxiv.org/pdf/1907.01743.pdf>.
    Uses two Daf3dResNetBottleneck blocks followed by two Daf3dResNetDilatedBottleneck blocks.

    Args:
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.

    """

    def __init__(
        self,
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: tuple[int] | int = 7,
        conv1_t_stride: tuple[int] | int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
    ):
        super().__init__(
            ResNetBottleneck,
            layers,
            block_inplanes,
            spatial_dims,
            n_input_channels,
            conv1_t_size,
            conv1_t_stride,
            no_max_pool,
            shortcut_type,
            widen_factor,
            num_classes,
            feed_forward,
            bias_downsample,
        )

        self.in_planes = 64

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.GROUP, spatial_dims]

        # adapt first convolution to work with new in_planes
        self.conv1 = conv_type(
            n_input_channels, self.in_planes, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = norm_type(32, 64)
        self.relu = nn.PReLU()

        # adapt layers to our needs
        self.layer1 = self._make_layer(Daf3dResNetBottleneck, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(
            Daf3dResNetBottleneck,
            block_inplanes[1],
            layers[1],
            spatial_dims,
            shortcut_type,
            stride=(1, 2, 2),  # type: ignore
        )
        self.layer3 = self._make_layer(
            Daf3dResNetDilatedBottleneck, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=1
        )
        self.layer4 = self._make_layer(
            Daf3dResNetDilatedBottleneck, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=1
        )


class Daf3dBackbone(nn.Module):
    """
    Backbone for 3D Feature Pyramid Network in DAF3D module based on 'Deep Attentive Features for Prostate Segmentation in
    3D Transrectal Ultrasound' <https://arxiv.org/pdf/1907.01743.pdf>.

    Args:
        n_input_channels: number of input channels for the first convolution.
    """

    def __init__(self, n_input_channels):
        super().__init__()
        net = Daf3dResNet(
            layers=[3, 4, 6, 3],
            block_inplanes=[128, 256, 512, 1024],
            n_input_channels=n_input_channels,
            num_classes=2,
            bias_downsample=False,
        )
        net_modules = list(net.children())
        self.layer0 = nn.Sequential(*net_modules[:3])
        self.layer1 = nn.Sequential(*net_modules[3:5])
        self.layer2 = net_modules[5]
        self.layer3 = net_modules[6]
        self.layer4 = net_modules[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


class Daf3dFPN(FeaturePyramidNetwork):
    """
    Feature Pyramid Network as used in 'Deep Attentive Features for Prostate Segmentation in 3D Transrectal Ultrasound'
    <https://arxiv.org/pdf/1907.01743.pdf>.
    Omits 3x3x3 convolution of layer_blocks and interpolates resulting feature maps to be the same size as
    feature map with highest resolution.

    Args:
        spatial_dims: 2D or 3D images
        in_channels_list: number of channels for each feature map that is passed to the module
        out_channels: number of channels of the FPN representation
        extra_blocks: if provided, extra operations will be performed.
            It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels_list: list[int],
        out_channels: int,
        extra_blocks: ExtraFPNBlock | None = None,
    ):
        super().__init__(spatial_dims, in_channels_list, out_channels, extra_blocks)

        self.inner_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = Convolution(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                adn_ordering="NA",
                act="PRELU",
                norm=("group", {"num_groups": 32, "num_channels": 128}),
            )
            self.inner_blocks.append(inner_block_module)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x_values: list[Tensor] = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x_values[-1], -1)
        results = []
        results.append(last_inner)

        for idx in range(len(x_values) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x_values[idx], idx)
            feat_shape = inner_lateral.shape[2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="trilinear")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, last_inner)

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x_values, names)

        # bring all layers to same size
        results = [results[0]] + [F.interpolate(l, size=x["feat1"].size()[2:], mode="trilinear") for l in results[1:]]
        # make it back an OrderedDict
        out = OrderedDict(list(zip(names, results)))

        return out


class Daf3dBackboneWithFPN(BackboneWithFPN):
    """
    Same as BackboneWithFPN but uses custom Daf3DFPN as feature pyramid network

    Args:
        backbone: backbone network
        return_layers: a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list: number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels: number of channels in the FPN.
        spatial_dims: 2D or 3D images
        extra_blocks: if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: dict[str, str],
        in_channels_list: list[int],
        out_channels: int,
        spatial_dims: int | None = None,
        extra_blocks: ExtraFPNBlock | None = None,
    ) -> None:
        super().__init__(backbone, return_layers, in_channels_list, out_channels, spatial_dims, extra_blocks)

        if spatial_dims is None:
            if hasattr(backbone, "spatial_dims") and isinstance(backbone.spatial_dims, int):
                spatial_dims = backbone.spatial_dims
            elif isinstance(backbone.conv1, nn.Conv2d):
                spatial_dims = 2
            elif isinstance(backbone.conv1, nn.Conv3d):
                spatial_dims = 3
            else:
                raise ValueError(
                    "Could not determine value of  `spatial_dims` from backbone, please provide explicit value."
                )

        self.fpn = Daf3dFPN(spatial_dims, in_channels_list, out_channels, extra_blocks)


class DAF3D(nn.Module):
    """
    DAF3D network based on 'Deep Attentive Features for Prostate Segmentation in 3D Transrectal Ultrasound'
    <https://arxiv.org/pdf/1907.01743.pdf>.
    The network consists of a 3D Feature Pyramid Network which is applied on the feature maps of a 3D ResNet,
    followed by a custom Attention Module and an ASPP module.
    During training the supervised signal consists of the outputs of the FPN (four Single Layer Features, SLFs),
    the outputs of the attention module (four Attentive Features) and the final prediction.
    They are individually compared to the ground truth, the final loss consists of a weighted sum of all
    individual losses (see DAF3D tutorial for details).
    There is an additional possiblity to return all supervised signals as well as the Attentive Maps in validation
    mode to visualize inner functionality of the network.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        visual_output: whether to return all SLFs, Attentive Maps, Refined SLFs in validation mode
            can be used to visualize inner functionality of the network
    """

    def __init__(self, in_channels, out_channels, visual_output=False):
        super().__init__()
        self.visual_output = visual_output
        self.backbone_with_fpn = Daf3dBackboneWithFPN(
            backbone=Daf3dBackbone(in_channels),
            return_layers={"layer1": "feat1", "layer2": "feat2", "layer3": "feat3", "layer4": "feat4"},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=128,
            spatial_dims=3,
        )
        self.predict1 = nn.Conv3d(128, out_channels, kernel_size=1)

        group_norm = ("group", {"num_groups": 32, "num_channels": 64})
        act_prelu = ("prelu", {"num_parameters": 1, "init": 0.25})
        self.fuse = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=512,
                out_channels=64,
                kernel_size=1,
                adn_ordering="NA",
                norm=group_norm,
                act=act_prelu,
            ),
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                adn_ordering="NA",
                padding=1,
                norm=group_norm,
                act=act_prelu,
            ),
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                adn_ordering="NA",
                padding=1,
                norm=group_norm,
                act=act_prelu,
            ),
        )
        self.attention = AttentionModule(
            spatial_dims=3, in_channels=192, out_channels=64, norm=group_norm, act=act_prelu
        )

        self.refine = Convolution(3, 256, 64, kernel_size=1, adn_ordering="NA", norm=group_norm, act=act_prelu)
        self.predict2 = nn.Conv3d(64, out_channels, kernel_size=1)
        self.aspp = Daf3dASPP(
            spatial_dims=3,
            in_channels=64,
            conv_out_channels=64,
            out_channels=64,
            kernel_sizes=(3, 3, 3, 3),
            dilations=((1, 1, 1), (1, 6, 6), (1, 12, 12), (1, 18, 18)),  # type: ignore
            norm_type=group_norm,
            acti_type=None,
            bias=True,
        )

    def forward(self, x):
        # layers from 1 - 4
        single_layer_features = list(self.backbone_with_fpn(x).values())

        # first 4 supervised signals (SLFs 1 - 4)
        supervised1 = [self.predict1(slf) for slf in single_layer_features]

        mlf = self.fuse(torch.cat(single_layer_features, 1))

        attentive_features_maps = [self.attention(slf, mlf) for slf in single_layer_features]
        att_features, att_maps = tuple(zip(*attentive_features_maps))

        # second 4 supervised signals (af 1 - 4)
        supervised2 = [self.predict2(af) for af in att_features]

        # attentive maps as optional additional output
        supervised3 = [self.predict2(am) for am in att_maps]

        attentive_mlf = self.refine(torch.cat(att_features, 1))

        aspp = self.aspp(attentive_mlf)

        supervised_final = self.predict2(aspp)

        if self.training:
            output = supervised1 + supervised2 + [supervised_final]
            output = [F.interpolate(o, size=x.size()[2:], mode="trilinear") for o in output]
        else:
            if self.visual_output:
                supervised_final = F.interpolate(supervised_final, size=x.size()[2:], mode="trilinear")
                supervised_inner = [
                    F.interpolate(o, size=x.size()[2:], mode="trilinear")
                    for o in supervised1 + supervised2 + supervised3
                ]
                output = [supervised_final] + supervised_inner
            else:
                output = F.interpolate(supervised_final, size=x.size()[2:], mode="trilinear")
        return output
