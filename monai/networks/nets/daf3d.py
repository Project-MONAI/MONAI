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

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.aspp import Daf3dASPP
from monai.networks.blocks.backbone_fpn_utils import Daf3dBackboneWithFPN
from monai.networks.blocks.convolutions import Convolution
from monai.networks.nets.resnet import Daf3dResNet

__all__ = ["AttentionModule", "Daf3dBackbone", "DAF3D"]


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
            dilations=((1, 1, 1), (1, 6, 6), (1, 12, 12), (1, 18, 18)),
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
        att_features, att_maps = zip(*attentive_features_maps)

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
                output = supervised_final, supervised_inner
            else:
                output = F.interpolate(supervised_final, size=x.size()[2:], mode="trilinear")
        return output
