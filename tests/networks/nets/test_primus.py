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
from monai.networks.nets import Primus, PrimusS, create_primus
from tests.test_utils import dict_product

device = "cuda" if torch.cuda.is_available() else "cpu"

# small transformer so the tests stay fast; head_dim = 24 is divisible by 2 * spatial_dims for 2D and 3D
SMALL = {"embed_dim": 48, "num_layers": 2, "num_heads": 2, "channels_per_level": (4, 8, 8, 16)}

TEST_CASE_PRIMUS = [
    [
        {**SMALL, **params, "in_channels": 2, "out_channels": 3, "img_size": (16, 24, 32)[: params["spatial_dims"]]},
        (2, 2, *(16, 24, 32)[: params["spatial_dims"]]),
        (2, 3, *(16, 24, 32)[: params["spatial_dims"]]),
    ]
    for params in dict_product(
        spatial_dims=[2, 3],
        num_register_tokens=[0, 2],
        use_rope=[True, False],
        use_abs_pos_embed=[True, False],
        add_skips=[True, False],
    )
]


class TestPrimus(unittest.TestCase):
    @parameterized.expand(TEST_CASE_PRIMUS)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = Primus(**input_param).to(device)
        with eval_mode(net):
            result, mask = net(torch.randn(input_shape).to(device), return_mask=True)
        self.assertEqual(result.shape, expected_shape)
        self.assertIsNone(mask)

    def test_patch_drop(self):
        net = Primus(1, 2, 32, patch_drop_rate=0.75, num_register_tokens=2, **SMALL).to(device)
        net.train()
        x = torch.randn(2, 1, 32, 32, 32, device=device)
        out, mask = net(x, return_mask=True)
        self.assertEqual(out.shape, (2, 2, 32, 32, 32))
        self.assertEqual(mask.shape, (2, 1, 32, 32, 32))
        self.assertEqual(mask.dtype, torch.bool)
        # 64 tokens of 8^3 voxels, 16 of them kept per sample
        self.assertEqual(mask.flatten(1).sum(1).tolist(), [16 * 8**3] * 2)
        # the mask is constant within each patch
        patches = mask.view(2, 4, 8, 4, 8, 4, 8)
        self.assertTrue(torch.equal(patches.amin((2, 4, 6)), patches.amax((2, 4, 6))))
        out.sum().backward()
        self.assertIsNotNone(net.register_tokens.grad)
        with eval_mode(net):
            _, mask = net(x, return_mask=True)
        self.assertIsNone(mask)

    def test_depth_per_level(self):
        net = Primus(1, 2, (8, 16, 4), depth_per_level=(2, 1), **{**SMALL, "channels_per_level": (4, 8, 16)})
        self.assertEqual(net.patch_size, (4, 4, 4))
        self.assertEqual(net.grid_size, (2, 4, 1))
        with eval_mode(net):
            self.assertEqual(net(torch.randn(1, 1, 8, 16, 4)).shape, (1, 2, 8, 16, 4))

    def test_variant(self):
        net = PrimusS(in_channels=1, out_channels=2, img_size=16)
        self.assertEqual(len(net.blocks), 12)
        self.assertEqual(net.norm.normalized_shape, (396,))
        with self.assertRaises(ValueError):
            create_primus("XL", in_channels=1, out_channels=2, img_size=16)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            Primus(1, 2, 20, **SMALL)  # not divisible by the patch size
        with self.assertRaises(ValueError):
            Primus(1, 2, 16, **{**SMALL, "num_heads": 5})
        with self.assertRaises(ValueError):
            Primus(1, 2, 16, **{**SMALL, "num_heads": 3})  # rope needs head_dim 16 divisible by 2 * spatial_dims
        with self.assertRaises(ValueError):
            Primus(1, 2, 16, patch_drop_rate=1.0, **SMALL)
        net = Primus(1, 2, 16, **SMALL)
        with self.assertRaises(ValueError):
            net(torch.randn(1, 1, 32, 32, 32))


if __name__ == "__main__":
    unittest.main()
