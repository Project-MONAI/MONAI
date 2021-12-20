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

from typing import Dict, Optional, Union, cast

import torch
import torch.nn as nn

from monai.utils.module import optional_import

models, _ = optional_import("torchvision.models")


class MILModel(nn.Module):
    """
    Multiple Instance Learning (MIL) model, with a backbone classification model.
    Currently, it only works for 2D images, a typical use case is for classification of the
    digital pathology whole slide images. The expected shape of input data is `[B, N, C, H, W]`,
    where `B` is the batch_size of PyTorch Dataloader and `N` is the number of instances
    extracted from every original image in the batch. A tutorial example is available at:
    https://github.com/Project-MONAI/tutorials/tree/master/pathology/multiple_instance_learning.

    Args:
        num_classes: number of output classes.
        mil_mode: MIL algorithm, available values (Defaults to ``"att"``):

            - ``"mean"`` - average features from all instances, equivalent to pure CNN (non MIL).
            - ``"max"`` - retain only the instance with the max probability for loss calculation.
            - ``"att"`` - attention based MIL https://arxiv.org/abs/1802.04712.
            - ``"att_trans"`` - transformer MIL https://arxiv.org/abs/2111.01556.
            - ``"att_trans_pyramid"`` - transformer pyramid MIL https://arxiv.org/abs/2111.01556.

        pretrained: init backbone with pretrained weights, defaults to ``True``.
        backbone: Backbone classifier CNN (either ``None``, a ``nn.Module`` that returns features,
            or a string name of a torchvision model).
            Defaults to ``None``, in which case ResNet50 is used.
        backbone_num_features: Number of output features of the backbone CNN
            Defaults to ``None`` (necessary only when using a custom backbone)
        trans_blocks: number of the blocks in `TransformEncoder` layer.
        trans_dropout: dropout rate in `TransformEncoder` layer.

    """

    def __init__(
        self,
        num_classes: int,
        mil_mode: str = "att",
        pretrained: bool = True,
        backbone: Optional[Union[str, nn.Module]] = None,
        backbone_num_features: Optional[int] = None,
        trans_blocks: int = 4,
        trans_dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if num_classes <= 0:
            raise ValueError("Number of classes must be positive: " + str(num_classes))

        if mil_mode.lower() not in ["mean", "max", "att", "att_trans", "att_trans_pyramid"]:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.mil_mode = mil_mode.lower()
        self.attention = nn.Sequential()
        self.transformer = None  # type: Optional[nn.Module]

        if backbone is None:

            net = models.resnet50(pretrained=pretrained)
            nfc = net.fc.in_features  # save the number of final features
            net.fc = torch.nn.Identity()  # remove final linear layer

            self.extra_outputs = {}  # type: Dict[str, torch.Tensor]

            if mil_mode == "att_trans_pyramid":
                # register hooks to capture outputs of intermediate layers
                def forward_hook(layer_name):
                    def hook(module, input, output):
                        self.extra_outputs[layer_name] = output

                    return hook

                net.layer1.register_forward_hook(forward_hook("layer1"))
                net.layer2.register_forward_hook(forward_hook("layer2"))
                net.layer3.register_forward_hook(forward_hook("layer3"))
                net.layer4.register_forward_hook(forward_hook("layer4"))

        elif isinstance(backbone, str):

            # assume torchvision model string is provided
            torch_model = getattr(models, backbone, None)
            if torch_model is None:
                raise ValueError("Unknown torch vision model" + str(backbone))
            net = torch_model(pretrained=pretrained)

            if getattr(net, "fc", None) is not None:
                nfc = net.fc.in_features  # save the number of final features
                net.fc = torch.nn.Identity()  # remove final linear layer
            else:
                raise ValueError(
                    "Unable to detect FC layer for the torchvision model " + str(backbone),
                    ". Please initialize the backbone model manually.",
                )

        elif isinstance(backbone, nn.Module):
            # use a custom backbone
            net = backbone
            nfc = backbone_num_features

            if backbone_num_features is None:
                raise ValueError("Number of endencoder features must be provided for a custom backbone model")

        else:
            raise ValueError("Unsupported backbone")

        if backbone is not None and mil_mode not in ["mean", "max", "att", "att_trans"]:
            raise ValueError("Custom backbone is not supported for the mode:" + str(mil_mode))

        if self.mil_mode in ["mean", "max"]:
            pass
        elif self.mil_mode == "att":
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode == "att_trans":
            transformer = nn.TransformerEncoderLayer(d_model=nfc, nhead=8, dropout=trans_dropout)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=trans_blocks)
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode == "att_trans_pyramid":

            transformer_list = nn.ModuleList(
                [
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout), num_layers=trans_blocks
                    ),
                    nn.Sequential(
                        nn.Linear(768, 256),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.Sequential(
                        nn.Linear(1280, 256),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=2304, nhead=8, dropout=trans_dropout),
                        num_layers=trans_blocks,
                    ),
                ]
            )
            self.transformer = transformer_list
            nfc = nfc + 256
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        else:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.myfc = nn.Linear(nfc, num_classes)
        self.net = net

    def calc_head(self, x: torch.Tensor) -> torch.Tensor:

        sh = x.shape

        if self.mil_mode == "mean":
            x = self.myfc(x)
            x = torch.mean(x, dim=1)

        elif self.mil_mode == "max":
            x = self.myfc(x)
            x, _ = torch.max(x, dim=1)

        elif self.mil_mode == "att":

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        elif self.mil_mode == "att_trans" and self.transformer is not None:

            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        elif self.mil_mode == "att_trans_pyramid" and self.transformer is not None:

            l1 = torch.mean(self.extra_outputs["layer1"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l2 = torch.mean(self.extra_outputs["layer2"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l3 = torch.mean(self.extra_outputs["layer3"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l4 = torch.mean(self.extra_outputs["layer4"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)

            transformer_list = cast(nn.ModuleList, self.transformer)

            x = transformer_list[0](l1)
            x = transformer_list[1](torch.cat((x, l2), dim=2))
            x = transformer_list[2](torch.cat((x, l3), dim=2))
            x = transformer_list[3](torch.cat((x, l4), dim=2))

            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

        return x

    def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:

        sh = x.shape
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])

        x = self.net(x)
        x = x.reshape(sh[0], sh[1], -1)

        if not no_head:
            x = self.calc_head(x)

        return x
