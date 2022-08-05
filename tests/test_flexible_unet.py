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
from typing import TYPE_CHECKING

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import FlexibleUNet
from monai.utils import optional_import
from tests.utils import skip_if_downloading_fails

if TYPE_CHECKING:
    import torchvision

    has_torchvision = True
else:
    torchvision, has_torchvision = optional_import("torchvision")

if TYPE_CHECKING:
    import PIL

    has_pil = True
else:
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
    input_shape=224,
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
# only test the models B0, B3, B7
SEL_MODELS = [get_model_names()[i] for i in [0, 3, 7]]

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
# 128
# 2D 128x128 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False, True],
        in_channels=3,
        num_classes=10,
        input_shape=128,
    )
)
# 2D 256x256 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False, True],
        in_channels=3,
        num_classes=10,
        input_shape=256,
    )
)

# 3D 128x128x128 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False],
        in_channels=3,
        num_classes=10,
        input_shape=128,
    )
)

# 3D 256x256x256 input
CASES_VARIATIONS.extend(
    make_shape_cases(
        models=SEL_MODELS,
        spatial_dims=[2],
        batches=[1],
        pretrained=[False],
        in_channels=3,
        num_classes=10,
        input_shape=256,
    )
)


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


if __name__ == "__main__":
    unittest.main()
