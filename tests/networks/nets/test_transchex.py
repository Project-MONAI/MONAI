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

import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.transchex import Transchex
from tests.test_utils import dict_product, skip_if_downloading_fails, skip_if_quick

TEST_CASE_TRANSCHEX = [
    [
        {
            **{k: v for k, v in params.items() if k != "img_size"},
            "img_size": (params["img_size"],) * 2,
            "patch_size": (params["patch_size"],) * 2,
        },
        (2, params["num_classes"]),
    ]
    for params in dict_product(
        drop_out=[0.4],
        img_size=[224],
        in_channels=[3],
        num_classes=[8],
        num_language_layers=[2],
        num_mixed_layers=[3],
        num_vision_layers=[4],
        patch_size=[16, 32],
    )
]


@skip_if_quick
class TestTranschex(unittest.TestCase):
    @parameterized.expand(TEST_CASE_TRANSCHEX)
    def test_shape(self, input_param, expected_shape):
        with skip_if_downloading_fails():
            net = Transchex(**input_param)
        with eval_mode(net):
            result = net(torch.randint(2, (2, 512)), torch.randint(2, (2, 512)), torch.randn((2, 3, 224, 224)))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            Transchex(
                in_channels=3,
                img_size=(128, 128),
                patch_size=(16, 16),
                num_language_layers=2,
                num_mixed_layers=4,
                num_vision_layers=2,
                num_classes=2,
                drop_out=5.0,
            )

        with self.assertRaises(ValueError):
            Transchex(
                in_channels=1,
                img_size=(97, 97),
                patch_size=(16, 16),
                num_language_layers=6,
                num_mixed_layers=6,
                num_vision_layers=8,
                num_classes=8,
                drop_out=0.4,
            )


if __name__ == "__main__":
    unittest.main()
