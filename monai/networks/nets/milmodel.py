from enum import Enum
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from monai.utils.module import optional_import

models, _ = optional_import("torchvision.models")


class MilMode(Enum):
    MEAN = "mean"
    MAX = "max"
    ATT = "att"
    ATT_TRANS = "att_trans"
    ATT_TRANS_PYRAMID = "att_trans_pyramid"


class MILModel(nn.Module):
    """
    A wrapper around backbone classification model suitable for MIL

    Args:
        num_classes: number of output classes
        mil_mode: MIL variant (supported max, mean, att, att_trans, att_trans_pyramid
        pretrained: init backbone with pretrained weights. Defaults to True.
        backbone: Backbone classifier CNN. Defaults to None, it which case ResNet50 will be used.
        backbone_nfeatures: Number of output featues of the backbone CNN (necessary only when using custom backbone)

    mil_mode:
        MilMode.MEAN - average features from all instances, equivalent to pure CNN (non MIL)
        MilMode.MAX - retain only the instance with the max probability for loss calculation
        MilMode.ATT - attention based MIL https://arxiv.org/abs/1802.04712
        MilMode.ATT_TRANS - transformer MIL https://arxiv.org/abs/2111.01556
        MilMode.ATT_TRANS_PYRAMID - transformer pyramid MIL https://arxiv.org/abs/2111.01556

    """

    def __init__(
        self,
        num_classes: int,
        mil_mode: MilMode = MilMode.ATT,
        pretrained: bool = True,
        backbone: Optional[Union[str, nn.Module]] = None,
        backbone_nfeatures: Optional[int] = None,
        trans_blocks: int = 4,
        trans_dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if num_classes <= 0:
            raise ValueError("Number of classes must be positive: " + str(num_classes))

        self.mil_mode = mil_mode
        print("MILModel with mode", mil_mode, "num_classes", num_classes)
        self.attention = nn.Sequential()
        self.transformer = None  # type: Optional[nn.Module]

        if backbone is None:

            net = models.resnet50(pretrained=pretrained)
            nfc = net.fc.in_features  # save the number of final features
            net.fc = torch.nn.Identity()  # remove final linear layer

            self.extra_outputs = {}  # type: Dict[str, torch.Tensor]

            if mil_mode == MilMode.ATT_TRANS_PYRAMID:
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
            trch_model = getattr(models, backbone, None)
            if trch_model is None:
                raise ValueError("Unknown torch vision model" + str(backbone))
            net = trch_model(pretrained=pretrained)

            if getattr(net, "fc", None) is not None:
                nfc = net.fc.in_features  # save the number of final features
                net.fc = torch.nn.Identity()  # remove final linear layer
            else:
                raise ValueError(
                    "Unable to detect FC layer for torch vision model " + str(backbone),
                    ". Please initialize the backbone model manually.",
                )

        else:
            # use a custom backbone (untested)
            net = backbone
            nfc = backbone_nfeatures

            if backbone_nfeatures is None:
                raise ValueError("Number of endencoder features must be provided for a custom backbone model")

        if backbone is not None and mil_mode not in [MilMode.MEAN, MilMode.MAX, MilMode.ATT, MilMode.ATT_TRANS]:
            raise ValueError("Custom backbone is not supported for the mode:" + str(mil_mode))

        if self.mil_mode in [MilMode.MEAN, MilMode.MAX]:
            pass
        elif self.mil_mode == MilMode.ATT:
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode == MilMode.ATT_TRANS:
            transformer = nn.TransformerEncoderLayer(d_model=nfc, nhead=8, dropout=trans_dropout)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=trans_blocks)
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode == MilMode.ATT_TRANS_PYRAMID:

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

        if self.mil_mode == MilMode.MEAN:
            x = self.myfc(x)
            x = torch.mean(x, dim=1)

        elif self.mil_mode == MilMode.MAX:
            x = self.myfc(x)
            x, _ = torch.max(x, dim=1)

        elif self.mil_mode == MilMode.ATT:

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        elif self.mil_mode == MilMode.ATT_TRANS and self.transformer is not None:

            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        elif self.mil_mode == MilMode.ATT_TRANS_PYRAMID and self.transformer is not None:

            l1 = torch.mean(self.extra_outputs["layer1"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l2 = torch.mean(self.extra_outputs["layer2"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l3 = torch.mean(self.extra_outputs["layer3"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l4 = torch.mean(self.extra_outputs["layer4"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)

            transformer_list: List = self.transformer  # type: ignore
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
