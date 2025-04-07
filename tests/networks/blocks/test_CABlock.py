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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.cablock import CABlock, FeedForward
from monai.utils import optional_import
from tests.test_utils import SkipIfBeforePyTorchVersion, assert_allclose

einops, has_einops = optional_import("einops")


TEST_CASES_CAB = []
for spatial_dims in [2, 3]:
    for dim in [32, 64, 128]:
        for num_heads in [2, 4, 8]:
            for bias in [True, False]:
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "dim": dim,
                        "num_heads": num_heads,
                        "bias": bias,
                        "flash_attention": False,
                    },
                    (2, dim, *([16] * spatial_dims)),
                    (2, dim, *([16] * spatial_dims)),
                ]
                TEST_CASES_CAB.append(test_case)


TEST_CASES_FEEDFORWARD = [
    # Test different spatial dims, dimensions and expansion factors
    [{"spatial_dims": 2, "dim": 64, "ffn_expansion_factor": 2.0, "bias": True}, (2, 64, 32, 32)],
    [{"spatial_dims": 3, "dim": 128, "ffn_expansion_factor": 1.5, "bias": False}, (2, 128, 16, 16, 16)],
    [{"spatial_dims": 2, "dim": 256, "ffn_expansion_factor": 1.0, "bias": True}, (1, 256, 64, 64)],
]


class TestFeedForward(unittest.TestCase):

    @parameterized.expand(TEST_CASES_FEEDFORWARD)
    def test_shape(self, input_param, input_shape):
        net = FeedForward(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, input_shape)

    def test_gating_mechanism(self):
        net = FeedForward(spatial_dims=2, dim=32, ffn_expansion_factor=2.0, bias=True)
        x = torch.ones(1, 32, 16, 16)
        out = net(x)
        self.assertNotEqual(torch.sum(out), torch.sum(x))


class TestCABlock(unittest.TestCase):

    @parameterized.expand(TEST_CASES_CAB)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = CABlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_invalid_spatial_dims(self):
        with self.assertRaises(ValueError):
            CABlock(spatial_dims=4, dim=64, num_heads=4, bias=True)

    @SkipIfBeforePyTorchVersion((2, 0))
    @skipUnless(has_einops, "Requires einops")
    def test_flash_attention(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        block = CABlock(spatial_dims=2, dim=64, num_heads=4, bias=True, flash_attention=True).to(device)
        x = torch.randn(2, 64, 32, 32).to(device)
        output = block(x)
        self.assertEqual(output.shape, x.shape)

    @skipUnless(has_einops, "Requires einops")
    def test_temperature_parameter(self):
        block = CABlock(spatial_dims=2, dim=64, num_heads=4, bias=True)
        self.assertTrue(isinstance(block.temperature, torch.nn.Parameter))
        self.assertEqual(block.temperature.shape, (4, 1, 1))

    @skipUnless(has_einops, "Requires einops")
    def test_qkv_transformation_2d(self):
        block = CABlock(spatial_dims=2, dim=64, num_heads=4, bias=True)
        x = torch.randn(2, 64, 32, 32)
        qkv = block.qkv(x)
        self.assertEqual(qkv.shape, (2, 192, 32, 32))

    @skipUnless(has_einops, "Requires einops")
    def test_qkv_transformation_3d(self):
        block = CABlock(spatial_dims=3, dim=64, num_heads=4, bias=True)
        x = torch.randn(2, 64, 16, 16, 16)
        qkv = block.qkv(x)
        self.assertEqual(qkv.shape, (2, 192, 16, 16, 16))

    @SkipIfBeforePyTorchVersion((2, 0))
    @skipUnless(has_einops, "Requires einops")
    def test_flash_vs_normal_attention(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        block_flash = CABlock(spatial_dims=2, dim=64, num_heads=4, bias=True, flash_attention=True).to(device)
        block_normal = CABlock(spatial_dims=2, dim=64, num_heads=4, bias=True, flash_attention=False).to(device)

        block_normal.load_state_dict(block_flash.state_dict())

        x = torch.randn(2, 64, 32, 32).to(device)
        with torch.no_grad():
            out_flash = block_flash(x)
            out_normal = block_normal(x)

        assert_allclose(out_flash, out_normal, atol=1e-4)

    @skipUnless(has_einops, "Requires einops")
    def test_deterministic_small_input(self):
        block = CABlock(spatial_dims=2, dim=2, num_heads=1, bias=False)
        with torch.no_grad():
            block.qkv.conv.weight.data.fill_(1.0)
            block.qkv_dwconv.conv.weight.data.fill_(1.0)
            block.temperature.data.fill_(1.0)
            block.project_out.conv.weight.data.fill_(1.0)

        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=torch.float32)

        output = block(x)
        # Channel attention: sum([1..8]) * (qkv_conv=1) * (dwconv=1) * (attn_weights=1) * (proj=1) = 36 * 2 = 72
        expected = torch.full_like(x, 72.0)

        assert_allclose(output, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
