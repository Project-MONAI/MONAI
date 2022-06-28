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
from monai.networks.nets import HoverNet
from tests.utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [  # fast mode, batch 16
    {
        "out_classes":5,
        "mode": HoverNet.Mode.FAST
    },
    (16, 3, 256, 256),
    {"nucleus_prediction":(16, 2, 164, 164),"type_prediction":(16, 5, 164, 164),"horizonal_vertical":(16, 2, 164, 164)}
]

TEST_CASE_1 = [  # single channel 2D, batch 16
    {
        "out_classes": None,
        "mode": HoverNet.Mode.FAST
    },
    (16, 3, 256, 256),
    {"nucleus_prediction":(16, 2, 164, 164),"horizonal_vertical":(16, 2, 164, 164)}
]

TEST_CASE_2 = [  # single channel 3D, batch 16
    {
        "out_classes": None,
        "mode": HoverNet.Mode.ORIGINAL
    },
    (16, 3, 270, 270),
    {"nucleus_prediction":(16, 2, 80, 80),"horizonal_vertical":(16, 2, 80, 80)}
]

TEST_CASE_3 = [  # 4-channel 3D, batch 16
    {
        "out_classes": 6,
        "mode": HoverNet.Mode.ORIGINAL
    },
    (16, 3, 270, 270),
    {"nucleus_prediction":(16, 2, 80, 80),"type_prediction":(16, 6, 80, 80),"horizonal_vertical":(16, 2, 80, 80)}
]

TEST_CASE_4 = [  # 4-channel 3D, batch 16, batch normalization
    {
        "out_classes": None,
        "mode": HoverNet.Mode.FAST,
        "dropout_prob": 0.5
    },
    (16, 3, 256, 256),
    {"nucleus_prediction":(16, 2, 164, 164),"horizonal_vertical":(16, 2, 164, 164)},
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4]

ILL_CASES = [
    [
        {
            "out_classes": 6,
            "mode": 3
        }
    ],
    [
        {
            "out_classes": 1000,
            "mode": HoverNet.Mode.ORIGINAL
        }
    ],
    [
        {
            "out_classes": 1,
            "mode": HoverNet.Mode.ORIGINAL
        }
    ],
    [
        {
            "out_classes": 6,
            "mode": HoverNet.Mode.ORIGINAL,
            "dropout_prob": 100
        }
    ],
]


class TestHoverNet(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shapes):
        net = HoverNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            for item in result:
                self.assertEqual(result[item].shape, expected_shapes[item])


    def test_script(self):
        net = HoverNet(mode=HoverNet.Mode.FAST)
        test_data = torch.randn(16, 3, 256, 256)
        test_script_save(net, test_data)

    def test_script_without_running_stats(self):
        net = HoverNet(mode=HoverNet.Mode.FAST)
        test_data = torch.randn(16, 3, 256, 256)
        test_script_save(net, test_data)

    def test_ill_input_shape(self):
        net = HoverNet(mode=HoverNet.Mode.FAST)
        with eval_mode(net):
            with self.assertRaises(ValueError):
                net.forward(torch.randn(1, 3, 270, 260))

    @parameterized.expand(ILL_CASES)
    def test_ill_input_hyper_params(self, input_param):
        with self.assertRaises(ValueError):
            _ = HoverNet(**input_param)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
