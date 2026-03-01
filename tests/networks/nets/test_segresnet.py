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
from monai.networks.nets import SegResNet, SegResNetVAE
from monai.utils import UpsampleMode
from tests.test_utils import dict_product, test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_SEGRESNET = [
    [
        {**params, "use_conv_final": False},
        (2, 1, *([16] * params["spatial_dims"])),
        (2, params["init_filters"], *([16] * params["spatial_dims"])),
    ]
    for params in dict_product(
        spatial_dims=range(2, 4),
        init_filters=[8, 16],
        dropout_prob=[None, 0.2],
        norm=[("GROUP", {"num_groups": 8}), ("batch", {"track_running_stats": False}), "instance"],
        upsample_mode=list(UpsampleMode),
    )
]

TEST_CASE_SEGRESNET_2 = [
    [params, (2, 1, *([16] * params["spatial_dims"])), (2, params["out_channels"], *([16] * params["spatial_dims"]))]
    for params in dict_product(
        spatial_dims=range(2, 4), init_filters=[8, 16], out_channels=range(1, 3), upsample_mode=list(UpsampleMode)
    )
]

TEST_CASE_SEGRESNET_VAE = [
    [
        {
            **params,
            "act": ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            "input_image_size": ([16] * params["spatial_dims"]),
        },
        (2, 1, *([16] * params["spatial_dims"])),
        (2, params["out_channels"], *([16] * params["spatial_dims"])),
    ]
    for params in dict_product(
        spatial_dims=range(2, 4),
        init_filters=[8, 16],
        out_channels=range(1, 3),
        upsample_mode=list(UpsampleMode),
        vae_estimate_std=[True, False],
    )
]


class TestResNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SEGRESNET + TEST_CASE_SEGRESNET_2)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SegResNet(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SegResNet(spatial_dims=4)

    def test_script(self):
        input_param, input_shape, expected_shape = TEST_CASE_SEGRESNET[0]
        net = SegResNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


class TestResNetVAE(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SEGRESNET_VAE)
    def test_vae_shape(self, input_param, input_shape, expected_shape):
        net = SegResNetVAE(**input_param).to(device)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, expected_shape = TEST_CASE_SEGRESNET_VAE[0]
        net = SegResNetVAE(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
