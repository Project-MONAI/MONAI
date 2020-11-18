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

import unittest

import torch
from parameterized import parameterized

from monai.networks.nets import densenet121, densenet169, densenet201, densenet264
from tests.utils import skip_if_quick, test_pretrained_networks

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_1 = [  # 4-channel 3D, batch 2
    {"pretrained": False, "spatial_dims": 3, "in_channels": 2, "out_channels": 3},
    (2, 2, 32, 64, 48),
    (2, 3),
]

TEST_CASE_2 = [  # 4-channel 2D, batch 2
    {"pretrained": False, "spatial_dims": 2, "in_channels": 2, "out_channels": 3},
    (2, 2, 32, 64),
    (2, 3),
]

TEST_CASE_3 = [  # 4-channel 1D, batch 1
    {"pretrained": False, "spatial_dims": 1, "in_channels": 2, "out_channels": 3},
    (1, 2, 32),
    (1, 3),
]

TEST_CASES = []
for case in [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]:
    for model in [densenet121, densenet169, densenet201, densenet264]:
        TEST_CASES.append([model, *case])


TEST_PRETRAINED_2D_CASE_1 = [  # 4-channel 2D, batch 2
    densenet121,
    {"pretrained": True, "progress": True, "spatial_dims": 2, "in_channels": 2, "out_channels": 3},
    (2, 2, 32, 64),
    (2, 3),
]

TEST_PRETRAINED_2D_CASE_2 = [  # 4-channel 2D, batch 2
    densenet121,
    {"pretrained": True, "progress": False, "spatial_dims": 2, "in_channels": 2, "out_channels": 3},
    (2, 2, 32, 64),
    (2, 3),
]


class TestPretrainedDENSENET(unittest.TestCase):
    @parameterized.expand([TEST_PRETRAINED_2D_CASE_1, TEST_PRETRAINED_2D_CASE_2])
    @skip_if_quick
    def test_121_3d_shape_pretrain(self, model, input_param, input_shape, expected_shape):
        net = test_pretrained_networks(model, input_param, device)
        net.eval()
        with torch.no_grad():
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


class TestDENSENET(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_densenet_shape(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param).to(device)
        net.eval()
        with torch.no_grad():
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
