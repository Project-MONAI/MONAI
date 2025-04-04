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
from monai.networks.nets.restormer import MDTATransformerBlock, OverlapPatchEmbed, Restormer
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

TEST_CASES_TRANSFORMER = [
    # [spatial_dims, dim, num_heads, ffn_factor, bias, layer_norm_use_bias, flash_attn, input_shape]
    [2, 48, 8, 2.66, True, True, False, (2, 48, 64, 64)],
    [2, 96, 8, 2.66, False, False, False, (2, 96, 32, 32)],
    [3, 48, 4, 2.66, True, True, False, (2, 48, 32, 32, 32)],
    [3, 96, 8, 2.66, False, False, True, (2, 96, 16, 16, 16)],
]

TEST_CASES_PATCHEMBED = [
    # spatial_dims, in_channels, embed_dim, input_shape, expected_shape
    [2, 1, 48, (2, 1, 64, 64), (2, 48, 64, 64)],
    [2, 3, 96, (2, 3, 32, 32), (2, 96, 32, 32)],
    [3, 1, 48, (2, 1, 32, 32, 32), (2, 48, 32, 32, 32)],
    [3, 4, 64, (2, 4, 16, 16, 16), (2, 64, 16, 16, 16)],
]

RESTORMER_CONFIGS = [
    # 2-level architecture
    {"num_blocks": [1, 1], "heads": [1, 1]},
    {"num_blocks": [2, 1], "heads": [2, 1]},
    # 3-level architecture
    {"num_blocks": [1, 1, 1], "heads": [1, 1, 1]},
    {"num_blocks": [2, 1, 1], "heads": [2, 1, 1]},
]

TEST_CASES_RESTORMER = []
for config in RESTORMER_CONFIGS:
    # 2D cases
    TEST_CASES_RESTORMER.extend(
        [
            [
                {
                    "spatial_dims": 2,
                    "in_channels": 1,
                    "out_channels": 1,
                    "dim": 48,
                    "num_blocks": config["num_blocks"],
                    "heads": config["heads"],
                    "num_refinement_blocks": 2,
                    "ffn_expansion_factor": 1.5,
                },
                (2, 1, 64, 64),
                (2, 1, 64, 64),
            ],
            # 3D cases
            [
                {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "dim": 16,
                    "num_blocks": config["num_blocks"],
                    "heads": config["heads"],
                    "num_refinement_blocks": 2,
                    "ffn_expansion_factor": 1.5,
                },
                (2, 1, 32, 32, 32),
                (2, 1, 32, 32, 32),
            ],
        ]
    )


class TestMDTATransformerBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASES_TRANSFORMER)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, spatial_dims, dim, heads, ffn_factor, bias, layer_norm_use_bias, flash, shape):
        if flash and not torch.cuda.is_available():
            self.skipTest("Flash attention requires CUDA")
        block = MDTATransformerBlock(
            spatial_dims=spatial_dims,
            dim=dim,
            num_heads=heads,
            ffn_expansion_factor=ffn_factor,
            bias=bias,
            layer_norm_use_bias=layer_norm_use_bias,
            flash_attention=flash,
        )
        with eval_mode(block):
            x = torch.randn(shape)
            output = block(x)
            self.assertEqual(output.shape, x.shape)


class TestOverlapPatchEmbed(unittest.TestCase):

    @parameterized.expand(TEST_CASES_PATCHEMBED)
    def test_shape(self, spatial_dims, in_channels, embed_dim, input_shape, expected_shape):
        net = OverlapPatchEmbed(spatial_dims=spatial_dims, in_channels=in_channels, embed_dim=embed_dim)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)


class TestRestormer(unittest.TestCase):

    @parameterized.expand(TEST_CASES_RESTORMER)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        if input_param.get("flash_attention", False) and not torch.cuda.is_available():
            self.skipTest("Flash attention requires CUDA")
        net = Restormer(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_small_input_error_2d(self):
        net = Restormer(spatial_dims=2, in_channels=1, out_channels=1)
        with self.assertRaises(AssertionError):
            net(torch.randn(1, 1, 8, 8))

    @skipUnless(has_einops, "Requires einops")
    def test_small_input_error_3d(self):
        net = Restormer(spatial_dims=3, in_channels=1, out_channels=1)
        with self.assertRaises(AssertionError):
            net(torch.randn(1, 1, 8, 8, 8))


if __name__ == "__main__":
    unittest.main()
