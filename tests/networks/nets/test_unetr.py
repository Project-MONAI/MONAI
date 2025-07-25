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
from monai.networks.nets.unetr import UNETR
from tests.test_utils import SkipIfBeforePyTorchVersion, dict_product, skip_if_quick, test_script_save

TEST_CASE_UNETR = [
    [
        {
            **{k: v for k, v in params.items() if k not in ["img_size", "nd"]},
            "conv_block": True,
            "res_block": False,
            "img_size": (params["img_size"],) * params["nd"],
            **({"spatial_dims": 2} if params["nd"] == 2 else {}),
        },
        (2, params["in_channels"], *([params["img_size"]] * params["nd"])),
        (2, params["out_channels"], *([params["img_size"]] * params["nd"])),
    ]
    for params in dict_product(
        dropout_rate=[0.4],
        feature_size=[16],
        hidden_size=[768],
        img_size=[96, 128],
        in_channels=[1],
        mlp_dim=[3072],
        nd=[2, 3],
        norm_name=["instance"],
        num_heads=[8],
        out_channels=[2],
        proj_type=["perceptron"],
    )
]


@skip_if_quick
class TestUNETR(unittest.TestCase):
    @parameterized.expand(TEST_CASE_UNETR)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UNETR(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            UNETR(
                in_channels=1,
                out_channels=3,
                img_size=(128, 128, 128),
                feature_size=16,
                hidden_size=128,
                mlp_dim=3072,
                num_heads=12,
                proj_type="conv",
                norm_name="instance",
                dropout_rate=5.0,
            )

        with self.assertRaises(ValueError):
            UNETR(
                in_channels=1,
                out_channels=4,
                img_size=(32, 32, 32),
                feature_size=32,
                hidden_size=512,
                mlp_dim=3072,
                num_heads=12,
                proj_type="conv",
                norm_name="instance",
                dropout_rate=0.5,
            )

        with self.assertRaises(ValueError):
            UNETR(
                in_channels=1,
                out_channels=3,
                img_size=(96, 96, 96),
                feature_size=16,
                hidden_size=512,
                mlp_dim=3072,
                num_heads=14,
                proj_type="conv",
                norm_name="batch",
                dropout_rate=0.4,
            )

        with self.assertRaises(ValueError):
            UNETR(
                in_channels=1,
                out_channels=4,
                img_size=(96, 96, 96),
                feature_size=8,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                proj_type="perc",
                norm_name="instance",
                dropout_rate=0.2,
            )

    @parameterized.expand(TEST_CASE_UNETR)
    @SkipIfBeforePyTorchVersion((2, 0))
    def test_script(self, input_param, input_shape, _):
        net = UNETR(**(input_param))
        net.eval()
        with torch.no_grad():
            torch.jit.script(net)

        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
