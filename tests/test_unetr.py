# Copyright 2020 - 2021 MONAI Consortium
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
from monai.networks.nets.unetr import UNETR

TEST_CASE_UNETR = []
for dropout_rate in [0.4]:
    for in_channels in [1, 4]:
        for out_channels in [2, 3]:
            for hidden_size in [768]:
                for img_size in [96, 128]:
                    for feature_size in [16]:
                        for num_heads in [8]:
                            for mlp_dim in [3072]:
                                for norm_name in ["instance"]:
                                    for pos_embed in ["conv", "perceptron"]:
                                        for conv_block in [True]:
                                            for res_block in [True, False]:
                                                test_case = [
                                                    {
                                                        "in_channels": in_channels,
                                                        "out_channels": out_channels,
                                                        "img_size": (img_size, img_size, img_size),
                                                        "hidden_size": hidden_size,
                                                        "feature_size": feature_size,
                                                        "norm_name": norm_name,
                                                        "mlp_dim": mlp_dim,
                                                        "num_heads": num_heads,
                                                        "pos_embed": pos_embed,
                                                        "dropout_rate": dropout_rate,
                                                        "conv_block": conv_block,
                                                        "res_block": res_block,
                                                    },
                                                    (2, in_channels, img_size, *([img_size] * 2)),
                                                    (2, out_channels, img_size, *([img_size] * 2)),
                                                ]
                                                TEST_CASE_UNETR.append(test_case)


class TestPatchEmbeddingBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_UNETR)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UNETR(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(AssertionError):
            UNETR(
                in_channels=1,
                out_channels=3,
                img_size=(128, 128, 128),
                feature_size=16,
                hidden_size=128,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="conv",
                norm_name="instance",
                dropout_rate=5.0,
            )

        with self.assertRaises(AssertionError):
            UNETR(
                in_channels=1,
                out_channels=4,
                img_size=(32, 32, 32),
                feature_size=32,
                hidden_size=512,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="conv",
                norm_name="instance",
                dropout_rate=0.5,
            )

        with self.assertRaises(AssertionError):
            UNETR(
                in_channels=1,
                out_channels=3,
                img_size=(96, 96, 96),
                feature_size=16,
                hidden_size=512,
                mlp_dim=3072,
                num_heads=14,
                pos_embed="conv",
                norm_name="batch",
                dropout_rate=0.4,
            )

        with self.assertRaises(KeyError):
            UNETR(
                in_channels=1,
                out_channels=4,
                img_size=(96, 96, 96),
                feature_size=8,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perc",
                norm_name="instance",
                dropout_rate=0.2,
            )


if __name__ == "__main__":
    unittest.main()
