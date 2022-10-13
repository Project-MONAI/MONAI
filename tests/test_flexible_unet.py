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

import unittest
from typing import Dict, List, Tuple

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import BACKBONE, EfficientNetBNFeatures, FlexibleUNet, ResNet, ResNetBlock, ResNetBottleneck
from monai.networks.blocks.encoder import BasicEncoder
from monai.utils import optional_import
from tests.utils import skip_if_downloading_fails, skip_if_quick


torchvision, has_torchvision = optional_import("torchvision")
PIL, has_pil = optional_import("PIL")


class ResNetEncoder(ResNet, BasicEncoder):
    backbone_names = ["resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]
    output_feature_channels = [(64, 128, 256, 512)] * 3 + [(256, 512, 1024, 2048)] * 4
    parameter_layers = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 4, 6, 3],
        [3, 4, 6, 3],
        [3, 4, 23, 3],
        [3, 8, 36, 3],
        [3, 24, 36, 3],
    ]

    def __init__(self, in_channels, pretrained, **kargs):
        super().__init__(**kargs, n_input_channels=in_channels)
        if pretrained:
            # Author of paper zipped the state_dict on googledrive,
            # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
            # Would like to load dict from url but need somewhere to save the state dicts.
            raise NotImplementedError(
                "Currently not implemented. You need to manually download weights provided by the paper's author"
                " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
            )

    @staticmethod
    def get_inplanes():
        return [64, 128, 256, 512]

    @classmethod
    def get_backbone_parameter(cls) -> List[Dict]:
        """
        Get parameter list to initialize encoder networks.
        Each parameter dict must have `spatial_dims`, `in_channels`
        and `pretrained` parameters.
        """
        parameter_list = []
        for backbone in range(len(cls.backbone_names)):
            if backbone < 3:
                res_type = ResNetBlock
            else:
                res_type = ResNetBottleneck
            parameter_list.append(
                {
                    "block": res_type,
                    "layers": cls.parameter_layers[backbone],
                    "block_inplanes": ResNetEncoder.get_inplanes(),
                    "spatial_dims": 2,
                    "in_channels": 3,
                    "pretrained": False,
                }
            )
        return parameter_list

    @classmethod
    def get_output_feature_channel_list(cls) -> List[Tuple[int, ...]]:
        """
        Get number of output features' channel.
        """
        return cls.output_feature_channels

    @classmethod
    def get_output_feature_number_list(cls) -> List[int]:
        """
        Get number of output feature.
        """
        return [4] * 7

    @classmethod
    def get_encoder_name_string_list(cls) -> List[str]:
        """
        Get the name string of backbones which will be used to initialize flexible unet.
        """
        return cls.backbone_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        feature_list.append(x)
        x = self.layer2(x)
        feature_list.append(x)
        x = self.layer3(x)
        feature_list.append(x)
        x = self.layer4(x)
        feature_list.append(x)

        return feature_list


BACKBONE.regist_class(ResNetEncoder)


def get_model_names():
    return [f"efficientnet-b{d}" for d in range(8)]


def get_resnet_names():
    return ResNetEncoder.get_encoder_name_string_list()


def make_shape_cases(
    models,
    spatial_dims,
    batches,
    pretrained,
    in_channels=3,
    num_classes=10,
    input_shape=64,
    norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
):
    ret_tests = []
    for spatial_dim in spatial_dims:  # selected spatial_dims
        for batch in batches:  # check single batch as well as multiple batch input
            for model in models:  # selected models
                for is_pretrained in pretrained:  # pretrained or not pretrained
                    if ("resnet" in model) and is_pretrained:
                        continue
                    kwargs = {
                        "in_channels": in_channels,
                        "out_channels": num_classes,
                        "backbone": model,
                        "pretrained": is_pretrained,
                        "spatial_dims": spatial_dim,
                        "norm": norm,
                    }
                    ret_tests.append(
                        [
                            kwargs,
                            (batch, in_channels) + (input_shape,) * spatial_dim,
                            (batch, num_classes) + (input_shape,) * spatial_dim,
                        ]
                    )
    return ret_tests


# create list of selected models to speed up redundant tests
# only test the models B0, B3
SEL_MODELS = [get_model_names()[i] for i in [0, 3]]
SEL_MODELS += [get_resnet_names()[i] for i in [0, 1, 2]]

# pretrained=False cases
# 2D and 3D models are expensive so use selected models
CASES_2D = make_shape_cases(
    models=SEL_MODELS,
    spatial_dims=[2],
    batches=[1, 4],
    pretrained=[False],
    in_channels=3,
    num_classes=10,
    norm="instance",
)
CASES_3D = make_shape_cases(
    models=[SEL_MODELS[0]],
    spatial_dims=[3],
    batches=[1],
    pretrained=[False],
    in_channels=3,
    num_classes=10,
    norm="batch",
)

# varying num_classes and in_channels
CASES_VARIATIONS = []

# change num_classes test
# 20 classes
# 2D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=3, num_classes=20
    )
)
# 3D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=[SEL_MODELS[0]], spatial_dims=[3], batches=[1], pretrained=[False], in_channels=3, num_classes=20
    )
)

# change in_channels test
# 1 channel
# 2D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=1, num_classes=10
    )
)
# 8 channel
# 2D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS, spatial_dims=[2], batches=[1], pretrained=[False, True], in_channels=8, num_classes=10
    )
)
# 3D
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=[SEL_MODELS[0]], spatial_dims=[3], batches=[1], pretrained=[False], in_channels=1, num_classes=10
    )
)

# change input shape test
# 96
# 2D 96x96 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False, True],
        in_channels=3,
        num_classes=10,
        input_shape=96,
    )
)
# 2D 64x64 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False, True],
        in_channels=3,
        num_classes=10,
        input_shape=64,
    )
)

# 3D 32x32x32 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False],
        in_channels=3,
        num_classes=10,
        input_shape=32,
    )
)

# 3D 64x64x64 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False],
        in_channels=3,
        num_classes=10,
        input_shape=64,
    )
)

# pretrain weight verified
CASES_PRETRAIN = [
    (
        {
            "in_channels": 3,
            "out_channels": 10,
            "backbone": SEL_MODELS[0],
            "pretrained": True,
            "spatial_dims": 2,
            "norm": ("batch", {"eps": 1e-3, "momentum": 0.01}),
        },
        {
            "in_channels": 3,
            "num_classes": 10,
            "model_name": SEL_MODELS[0],
            "pretrained": True,
            "spatial_dims": 2,
            "norm": ("batch", {"eps": 1e-3, "momentum": 0.01}),
        },
        ["_conv_stem.weight"],
    )
]


@skip_if_quick
class TestFLEXIBLEUNET(unittest.TestCase):
    @parameterized.expand(CASES_2D + CASES_3D + CASES_VARIATIONS)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with skip_if_downloading_fails():
            net = FlexibleUNet(**input_param).to(device)

        # run inference with random tensor
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))

        # check output shape
        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(CASES_PRETRAIN)
    def test_pretrain(self, input_param, efficient_input_param, weight_list):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with skip_if_downloading_fails():
            net = FlexibleUNet(**input_param).to(device)

        with skip_if_downloading_fails():
            eff_net = EfficientNetBNFeatures(**efficient_input_param).to(device)

        for weight_name in weight_list:
            if weight_name in net.encoder.state_dict() and weight_name in eff_net.state_dict():
                net_weight = net.encoder.state_dict()[weight_name]
                download_weight = eff_net.state_dict()[weight_name]
                weight_diff = torch.abs(net_weight - download_weight)
                diff_sum = torch.sum(weight_diff)
                # check if a weight in weight_list equals to the downloaded weight.
                self.assertLess(abs(diff_sum.item() - 0), 1e-8)


if __name__ == "__main__":
    unittest.main()
