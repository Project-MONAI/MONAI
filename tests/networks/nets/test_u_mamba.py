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

from monai.networks.nets import MambaBlock, UMamba

TEST_CASES = [
    # 2D, bottleneck Mamba (U-Mamba_Bot)
    [
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "channels": (8, 16, 32), "strides": (1, 2, 2)},
        (2, 1, 32, 32),
        (2, 2, 32, 32),
    ],
    # 2D, Mamba after every encoder stage (U-Mamba_Enc)
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 2,
            "channels": (8, 16, 32),
            "strides": (1, 2, 2),
            "mamba_mode": "enc",
        },
        (1, 1, 32, 32),
        (1, 2, 32, 32),
    ],
    # 3D, bottleneck Mamba (U-Mamba_Bot)
    [
        {"spatial_dims": 3, "in_channels": 1, "out_channels": 3, "channels": (8, 16, 32), "strides": (1, 2, 2)},
        (1, 1, 16, 16, 16),
        (1, 3, 16, 16, 16),
    ],
    # 3D, Mamba after every encoder stage (U-Mamba_Enc)
    [
        {
            "spatial_dims": 3,
            "in_channels": 2,
            "out_channels": 2,
            "channels": (8, 16),
            "strides": (1, 2),
            "mamba_mode": "enc",
        },
        (1, 2, 8, 8, 8),
        (1, 2, 8, 8, 8),
    ],
    # 3D, anisotropic strides
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 2,
            "channels": (8, 16, 32),
            "strides": ((1, 1, 1), (2, 2, 2), (2, 2, 1)),
        },
        (1, 1, 16, 16, 8),
        (1, 2, 16, 16, 8),
    ],
    # 1D
    [
        {"spatial_dims": 1, "in_channels": 1, "out_channels": 2, "channels": (8, 16), "strides": (1, 2)},
        (1, 1, 64),
        (1, 2, 64),
    ],
    # per-stage kernel sizes and number of residual units
    [
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 2,
            "channels": (8, 16, 32),
            "strides": (1, 2, 2),
            "kernel_size": (3, 3, 5),
            "num_res_units": (2, 1, 2),
        },
        (1, 3, 32, 32),
        (1, 2, 32, 32),
    ],
]


class TestUMamba(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UMamba(**input_param)
        net.eval()
        with torch.no_grad():
            out = net(torch.randn(input_shape))
        self.assertEqual(out.shape, expected_shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_deep_supervision(self):
        net = UMamba(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(8, 16, 32),
            strides=(1, 2, 2),
            deep_supervision=True,
        )
        net.eval()
        with torch.no_grad():
            out = net(torch.randn(1, 1, 32, 32))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        # highest resolution first, then the native resolution of each deeper decoder stage
        self.assertEqual(out[0].shape, (1, 2, 32, 32))
        self.assertEqual(out[1].shape, (1, 2, 16, 16))

    def test_backward(self):
        net = UMamba(spatial_dims=3, in_channels=1, out_channels=2, channels=(8, 16, 32), strides=(1, 2, 2))
        net.train()
        out = net(torch.randn(1, 1, 16, 16, 16))
        out.mean().backward()
        mamba = net.bottleneck_mamba.mamba
        self.assertIsNotNone(mamba.A_log.grad)
        self.assertTrue(torch.isfinite(mamba.A_log.grad).all())
        encoder_grads = [p.grad for p in net.encoder.parameters() if p.grad is not None]
        self.assertTrue(len(encoder_grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in encoder_grads))

    def test_invalid_mamba_mode(self):
        with self.assertRaises(ValueError):
            UMamba(spatial_dims=2, mamba_mode="invalid")

    def test_mamba_block_shape(self):
        block = MambaBlock(dim=8)
        block.eval()
        with torch.no_grad():
            out = block(torch.randn(2, 8, 10, 10))
        self.assertEqual(out.shape, (2, 8, 10, 10))
        self.assertTrue(torch.isfinite(out).all())

    def test_mamba_block_half_precision(self):
        block = MambaBlock(dim=8)
        block.eval()
        with torch.no_grad():
            out = block(torch.randn(1, 8, 6, 6, dtype=torch.float16))
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, (1, 8, 6, 6))


if __name__ == "__main__":
    unittest.main()
