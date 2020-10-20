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

from monai.networks.nets import SegResNet, SegResNetVAE
from monai.utils import UpsampleMode

TEST_CASE_SEGRESNET = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for dropout_prob in [None, 0.2]:
            for norm_name in ["group", "batch", "instance"]:
                for upsample_mode in UpsampleMode:
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "init_filters": init_filters,
                            "dropout_prob": dropout_prob,
                            "norm_name": norm_name,
                            "upsample_mode": upsample_mode,
                            "use_conv_final": False,
                        },
                        torch.randn(2, 1, *([16] * spatial_dims)),
                        (2, init_filters, *([16] * spatial_dims)),
                    ]
                    TEST_CASE_SEGRESNET.append(test_case)

TEST_CASE_SEGRESNET_2 = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for out_channels in range(1, 3):
            for upsample_mode in UpsampleMode:
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "init_filters": init_filters,
                        "out_channels": out_channels,
                        "upsample_mode": upsample_mode,
                    },
                    torch.randn(2, 1, *([16] * spatial_dims)),
                    (2, out_channels, *([16] * spatial_dims)),
                ]
                TEST_CASE_SEGRESNET_2.append(test_case)

TEST_CASE_SEGRESNET_VAE = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for out_channels in range(1, 3):
            for upsample_mode in UpsampleMode:
                for vae_estimate_std in [True, False]:
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "init_filters": init_filters,
                            "out_channels": out_channels,
                            "upsample_mode": upsample_mode,
                            "input_image_size": ([16] * spatial_dims),
                            "vae_estimate_std": vae_estimate_std,
                        },
                        torch.randn(2, 1, *([16] * spatial_dims)),
                        (2, out_channels, *([16] * spatial_dims)),
                    ]
                TEST_CASE_SEGRESNET_VAE.append(test_case)


class TestResBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SEGRESNET + TEST_CASE_SEGRESNET_2)
    def test_shape(self, input_param, input_data, expected_shape):
        net = SegResNet(**input_param)
        net.eval()
        with torch.no_grad():
            result = net(input_data)
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(AssertionError):
            SegResNet(spatial_dims=4)


class TestResBlockVAE(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SEGRESNET_VAE)
    def test_vae_shape(self, input_param, input_data, expected_shape):
        net = SegResNetVAE(**input_param)
        with torch.no_grad():
            result, _ = net(input_data)
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
