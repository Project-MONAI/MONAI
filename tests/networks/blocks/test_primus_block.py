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
from monai.networks.blocks.primus_block import LayerNormNd, PrimusPatchDecode, PrimusPatchEmbed, ResidualBlockD

TEST_CASES_EMBED = [
    [3, (1, 1, 1), (8, 8, 16, 32), True, (2, 1, 16, 16, 16), (2, 24, 2, 2, 2)],
    [3, (2, 1), (4, 8, 8), False, (1, 2, 8, 12, 16), (1, 24, 2, 3, 4)],
    [2, (1, 1, 1), (8, 8, 16, 32), True, (2, 3, 32, 16), (2, 24, 4, 2)],
]

TEST_CASES_DECODE = [
    [3, (8, 8, 8), 96, 3, (2, 96, 2, 2, 2), (2, 3, 16, 16, 16)],
    [3, (8, 4, 2), 96, 2, (1, 96, 2, 3, 4), (1, 2, 16, 12, 8)],
    [2, (16, 16), 64, 5, (1, 64, 2, 3), (1, 5, 32, 48)],
    [3, (2, 2, 2), 32, 1, (1, 32, 3, 3, 3), (1, 1, 6, 6, 6)],
]


class TestPrimusBlocks(unittest.TestCase):
    @parameterized.expand(TEST_CASES_EMBED)
    def test_patch_embed(self, spatial_dims, depth_per_level, channels_per_level, add_skips, in_shape, out_shape):
        net = PrimusPatchEmbed(spatial_dims, in_shape[1], 24, depth_per_level, channels_per_level, add_skips)
        with eval_mode(net):
            self.assertEqual(net(torch.randn(in_shape)).shape, out_shape)

    @parameterized.expand(TEST_CASES_DECODE)
    def test_patch_decode(self, spatial_dims, patch_size, embed_dim, out_channels, in_shape, out_shape):
        net = PrimusPatchDecode(spatial_dims, patch_size, embed_dim, out_channels)
        with eval_mode(net):
            self.assertEqual(net(torch.randn(in_shape)).shape, out_shape)

    def test_residual_block(self):
        block = ResidualBlockD(3, 4, 8, stride=2)
        self.assertEqual(block(torch.randn(1, 4, 8, 8, 8)).shape, (1, 8, 4, 4, 4))
        self.assertEqual(len(ResidualBlockD(3, 8, 8).skip), 0)

    def test_layer_norm_nd(self):
        x = torch.randn(2, 6, 3, 4, 5)
        expected = torch.nn.functional.layer_norm(x.movedim(1, -1), (6,), eps=1e-6).movedim(-1, 1)
        torch.testing.assert_close(LayerNormNd(6)(x), expected)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            PrimusPatchEmbed(3, 1, 24, (1, 1), (8, 16))
        with self.assertRaises(ValueError):
            PrimusPatchEmbed(3, 1, 24, (1, 0), (8, 16, 32))
        with self.assertRaises(ValueError):
            PrimusPatchDecode(3, (8, 8, 6), 24, 2)
        with self.assertRaises(ValueError):
            PrimusPatchDecode(3, (8, 8), 24, 2)


if __name__ == "__main__":
    unittest.main()
