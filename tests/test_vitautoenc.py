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
from monai.networks.nets.vitautoenc import ViTAutoEnc

TEST_CASE_Vitautoenc = []
for in_channels in [1, 4]:
    for img_size in [64, 96, 128]:
        for patch_size in [16]:
            for pos_embed in ["conv", "perceptron"]:
                for nd in [2, 3]:
                    test_case = [
                        {
                            "in_channels": in_channels,
                            "img_size": (img_size,) * nd,
                            "patch_size": (patch_size,) * nd,
                            "hidden_size": 768,
                            "mlp_dim": 3072,
                            "num_layers": 4,
                            "num_heads": 12,
                            "pos_embed": pos_embed,
                            "dropout_rate": 0.6,
                            "spatial_dims": nd,
                        },
                        (2, in_channels, *([img_size] * nd)),
                        (2, 1, *([img_size] * nd)),
                    ]

                    TEST_CASE_Vitautoenc.append(test_case)

TEST_CASE_Vitautoenc.append(
    [
        {
            "in_channels": 1,
            "img_size": (512, 512, 32),
            "patch_size": (16, 16, 16),
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_layers": 4,
            "num_heads": 12,
            "pos_embed": "conv",
            "dropout_rate": 0.6,
            "spatial_dims": 3,
        },
        (2, 1, 512, 512, 32),
        (2, 1, 512, 512, 32),
    ]
)


class TestPatchEmbeddingBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_Vitautoenc)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = ViTAutoEnc(**input_param)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            ViTAutoEnc(
                in_channels=1,
                img_size=(128, 128, 128),
                patch_size=(16, 16, 16),
                hidden_size=128,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                pos_embed="conv",
                dropout_rate=5.0,
            )

        with self.assertRaises(ValueError):
            ViTAutoEnc(
                in_channels=1,
                img_size=(32, 32, 32),
                patch_size=(64, 64, 64),
                hidden_size=512,
                mlp_dim=3072,
                num_layers=12,
                num_heads=8,
                pos_embed="perceptron",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            ViTAutoEnc(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(8, 8, 8),
                hidden_size=512,
                mlp_dim=3072,
                num_layers=12,
                num_heads=14,
                pos_embed="conv",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            ViTAutoEnc(
                in_channels=1,
                img_size=(97, 97, 97),
                patch_size=(4, 4, 4),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=8,
                pos_embed="perceptron",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            ViTAutoEnc(
                in_channels=4,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                pos_embed="perc",
                dropout_rate=0.3,
            )


if __name__ == "__main__":
    unittest.main()
