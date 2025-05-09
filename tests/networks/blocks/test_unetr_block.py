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
from monai.networks.blocks.dynunet_block import get_padding
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from tests.test_utils import dict_product, test_script_save


def _get_out_size(params):
    in_size = params["in_size"]
    kernel_size = params["kernel_size"]
    stride = params["stride"]
    padding = get_padding(kernel_size, stride)
    if not isinstance(padding, int):
        padding = padding[0]
    return int((in_size + 2 * padding - kernel_size) / stride) + 1


TEST_CASE_UNETR_BASIC_BLOCK = [
    [
        {
            "spatial_dims": params["spatial_dims"],
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": params["kernel_size"],
            "norm_name": params["norm_name"],
            "stride": params["stride"],
        },
        (1, 16, *([params["in_size"]] * params["spatial_dims"])),
        (1, 16, *([_get_out_size(params)] * params["spatial_dims"]),),
    ]
    for params in dict_product(
        spatial_dims=range(1, 4),
        kernel_size=[1, 3],
        stride=[2],
        norm_name=[("GROUP", {"num_groups": 16}), ("batch", {"track_running_stats": False}), "instance"],
        in_size=[15, 16],
    )
]

TEST_UP_BLOCK = [
    [
        {
            "spatial_dims": params["spatial_dims"],
            "in_channels": params["in_channels"],
            "out_channels": params["out_channels"],
            "kernel_size": params["kernel_size"],
            "norm_name": params["norm_name"],
            "res_block": params["res_block"],
            "upsample_kernel_size": params["stride"],
        },
        (1, params["in_channels"], *([params["in_size"]] * params["spatial_dims"])),
        (1, params["out_channels"], *([params["in_size"] * params["stride"]] * params["spatial_dims"])),
        (1, params["out_channels"], *([params["in_size"] * params["stride"]] * params["spatial_dims"])),
    ]
    for params in dict_product(
        spatial_dims=range(1, 4),
        in_channels=[4],
        out_channels=[2],
        kernel_size=[1, 3],
        norm_name=["instance"],
        res_block=[False, True],
        upsample_kernel_size=[2, 3],
        stride=[1, 2],
        in_size=[15, 16],
    )
]

TEST_PRUP_BLOCK = []
in_channels, out_channels = 4, 2
for params in dict_product(
    spatial_dims=range(1, 4),
    kernel_size=[1, 3],
    upsample_kernel_size=[2, 3],
    stride=[1, 2],
    res_block=[False, True],
    norm_name=["instance"],
    in_size_scalar=[15, 16],
    num_layer=[0, 2],
):
    spatial_dims_val = params["spatial_dims"]
    in_size_val = params["in_size_scalar"]
    upsample_kernel_size_val = params["upsample_kernel_size"]
    num_layer_val = params["num_layer"]

    in_size_tmp = in_size_val
    out_size = 0  # Initialize out_size
    for _ in range(num_layer_val + 1):
        out_size = in_size_tmp * upsample_kernel_size_val
        in_size_tmp = out_size

    test_case = [
        {
            "spatial_dims": spatial_dims_val,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_layer": num_layer_val,
            "kernel_size": params["kernel_size"],
            "norm_name": params["norm_name"],
            "stride": params["stride"],
            "res_block": params["res_block"],
            "upsample_kernel_size": upsample_kernel_size_val,
        },
        (1, in_channels, *([in_size_val] * spatial_dims_val)),
        (1, out_channels, *([out_size] * spatial_dims_val)),
    ]
    TEST_PRUP_BLOCK.append(test_case)


class TestResBasicBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_UNETR_BASIC_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape):
        for net in [UnetrBasicBlock(**input_param)]:
            with eval_mode(net):
                result = net(torch.randn(input_shape))
                self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            UnetrBasicBlock(3, 4, 2, kernel_size=3, stride=1, norm_name="norm")
        with self.assertRaises(AssertionError):
            UnetrBasicBlock(3, 4, 2, kernel_size=1, stride=4, norm_name="batch")

    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_UNETR_BASIC_BLOCK[0]
        net = UnetrBasicBlock(**input_param)
        with eval_mode(net):
            test_data = torch.randn(input_shape)
            test_script_save(net, test_data)


class TestUpBlock(unittest.TestCase):
    @parameterized.expand(TEST_UP_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape, skip_shape):
        net = UnetrUpBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape), torch.randn(skip_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _, skip_shape = TEST_UP_BLOCK[0]
        net = UnetrUpBlock(**input_param)
        test_data = torch.randn(input_shape)
        skip_data = torch.randn(skip_shape)
        test_script_save(net, test_data, skip_data)


class TestPrUpBlock(unittest.TestCase):
    @parameterized.expand(TEST_PRUP_BLOCK)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UnetrPrUpBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _ = TEST_PRUP_BLOCK[0]
        net = UnetrPrUpBlock(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
