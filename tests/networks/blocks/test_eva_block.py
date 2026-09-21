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
from monai.networks.blocks.eva_block import EVAAttention, EVABlock, SwiGLUMLP
from monai.networks.blocks.rope import SpatialRotaryEmbedding

TEST_CASES = []
for scale_attn_inner in [True, False]:
    for init_values in [None, 0.1]:
        for num_prefix_tokens in [0, 2]:
            for use_rope in [True, False]:
                TEST_CASES.append([scale_attn_inner, init_values, num_prefix_tokens, use_rope])


class TestEVABlock(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, scale_attn_inner, init_values, num_prefix_tokens, use_rope):
        block = EVABlock(
            48,
            4,
            scale_attn_inner=scale_attn_inner,
            init_values=init_values,
            num_prefix_tokens=num_prefix_tokens,
            drop_path_rate=0.1,
        )
        rope = SpatialRotaryEmbedding(12, (2, 2, 2))() if use_rope else None
        x = torch.randn(2, 8 + num_prefix_tokens, 48)
        with eval_mode(block):
            self.assertEqual(block(x, rope=rope).shape, x.shape)

    def test_rope(self):
        attn = EVAAttention(16, 2, num_prefix_tokens=1)
        rope = SpatialRotaryEmbedding(8, (4,))()
        x = torch.randn(1, 5, 16)
        with eval_mode(attn):
            out_rope = attn(x, rope=rope)
            out_zero_angle = attn(x, rope=torch.cat([torch.zeros(4, 8), torch.ones(4, 8)], -1))
            out_plain = attn(x)
        torch.testing.assert_close(out_zero_angle, out_plain)
        self.assertFalse(torch.allclose(out_rope, out_plain))

    def test_layer_scale(self):
        block = EVABlock(16, 2, init_values=0.0)
        x = torch.randn(2, 4, 16)
        with eval_mode(block):
            torch.testing.assert_close(block(x), x)

    def test_swiglu(self):
        mlp = SwiGLUMLP(16, 40, scale_norm=False)
        self.assertEqual(mlp(torch.randn(3, 16)).shape, (3, 16))

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            EVAAttention(10, 3)
        for num_heads in (0, -2):
            with self.assertRaises(ValueError):
                EVAAttention(8, num_heads)


if __name__ == "__main__":
    unittest.main()
