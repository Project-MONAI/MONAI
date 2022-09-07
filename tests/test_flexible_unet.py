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

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import EfficientNetBNFeatures, FlexibleUNet
from monai.utils import optional_import
from tests.utils import skip_if_downloading_fails, skip_if_quick

torchvision, has_torchvision = optional_import("torchvision")
PIL, has_pil = optional_import("PIL")


def get_model_names():
    return [f"efficientnet-b{d}" for d in range(8)]


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
