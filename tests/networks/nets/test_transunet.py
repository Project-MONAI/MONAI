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
from monai.networks.nets.transunet import TransUNet
from tests.test_utils import dict_product, skip_if_quick

TEST_CASE_TRANsunet = [
    [
        {
            **{k: v for k, v in params.items() if k not in ["img_size", "nd"]},
            "img_size": (params["img_size"],) * params["nd"],
            "spatial_dims": params["nd"],
        },
        (2, params["in_channels"], *([params["img_size"]] * params["nd"])),
        (2, params["out_channels"], *([params["img_size"]] * params["nd"])),
    ]
    for params in dict_product(
        feature_size=[16],
        hidden_size=[192],
        in_channels=[1],
        mlp_dim=[768],
        nd=[2, 3],
        norm_name=["instance"],
        num_heads=[8],
        num_layers=[2],
        img_size=[64],
        out_channels=[2],
        dropout_rate=[0.4],
        res_block=[True],
    )
]


@skip_if_quick
class TestTransUNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_TRANsunet)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = TransUNet(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            TransUNet(in_channels=1, out_channels=3, img_size=(64, 64), dropout_rate=5.0)

        with self.assertRaises(ValueError):
            TransUNet(in_channels=1, out_channels=3, img_size=(64, 64), hidden_size=192, num_heads=7)

        with self.assertRaises(ValueError):
            TransUNet(in_channels=1, out_channels=3, img_size=(60, 60))


if __name__ == "__main__":
    unittest.main()
