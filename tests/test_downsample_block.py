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

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks import MaxAvgPool, SubpixelDownsample, SubpixelUpsample

TEST_CASES = [
    [{"spatial_dims": 2, "kernel_size": 2}, (7, 4, 64, 48), (7, 8, 32, 24)],  # 4-channel 2D, batch 7
    [{"spatial_dims": 1, "kernel_size": 4}, (16, 4, 63), (16, 8, 15)],  # 4-channel 1D, batch 16
    [{"spatial_dims": 1, "kernel_size": 4, "padding": 1}, (16, 4, 63), (16, 8, 16)],  # 4-channel 1D, batch 16
    [  # 4-channel 3D, batch 16
        {"spatial_dims": 3, "kernel_size": 3, "ceil_mode": True},
        (16, 4, 32, 24, 48),
        (16, 8, 11, 8, 16),
    ],
    [  # 1-channel 3D, batch 16
        {"spatial_dims": 3, "kernel_size": 3, "ceil_mode": False},
        (16, 1, 32, 24, 48),
        (16, 2, 10, 8, 16),
    ],
]

TEST_CASES_SUBPIXEL = [
    [{"spatial_dims": 2, "in_channels": 1, "scale_factor": 2}, (1, 1, 8, 8), (1, 4, 4, 4)],
    [{"spatial_dims": 3, "in_channels": 2, "scale_factor": 2}, (1, 2, 8, 8, 8), (1, 16, 4, 4, 4)],
    [{"spatial_dims": 1, "in_channels": 3, "scale_factor": 2}, (1, 3, 8), (1, 6, 4)],
]

class TestMaxAvgPool(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = MaxAvgPool(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)


class TestSubpixelDownsample(unittest.TestCase):
    
    @parameterized.expand(TEST_CASES_SUBPIXEL)
    def test_shape(self, input_param, input_shape, expected_shape):
        downsampler = SubpixelDownsample(**input_param)
        with eval_mode(downsampler):
            result = downsampler(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)
    
    def test_predefined_tensor(self):
        test_tensor = torch.arange(4).view(4, 1, 1).repeat(1, 4, 4)
        test_tensor = test_tensor.unsqueeze(0)

        downsampler = SubpixelDownsample(spatial_dims=2, in_channels=1, scale_factor=2, conv_block=None)
        with eval_mode(downsampler):
            result = downsampler(test_tensor)
            self.assertEqual(result.shape, (1, 16, 2, 2))
            self.assertTrue(torch.all(result[0, 0:3] == 0))
            self.assertTrue(torch.all(result[0, 4:7] == 1))
            self.assertTrue(torch.all(result[0, 8:11] == 2))
            self.assertTrue(torch.all(result[0, 12:15] == 3))

    def test_reconstruction_2D(self):
        input_tensor = torch.randn(1, 1, 4, 4)
        down = SubpixelDownsample(spatial_dims=2, in_channels=1, scale_factor=2, conv_block=None)
        up = SubpixelUpsample(spatial_dims=2, in_channels=4, scale_factor=2, conv_block=None, apply_pad_pool=False)
        with eval_mode(down), eval_mode(up):
            downsampled = down(input_tensor)
            reconstructed = up(downsampled)
            self.assertTrue(torch.allclose(input_tensor, reconstructed, rtol=1e-5))

    def test_reconstruction_3D(self):
        input_tensor = torch.randn(1, 1, 4, 4, 4)
        down = SubpixelDownsample(spatial_dims=3, in_channels=1, scale_factor=2, conv_block=None)
        up = SubpixelUpsample(spatial_dims=3, in_channels=4, scale_factor=2, conv_block=None, apply_pad_pool=False)
        with eval_mode(down), eval_mode(up):
            downsampled = down(input_tensor)
            reconstructed = up(downsampled)
            self.assertTrue(torch.allclose(input_tensor, reconstructed, rtol=1e-5))


    def test_invalid_spatial_size(self):
        downsampler = SubpixelDownsample(spatial_dims=2, in_channels=1, scale_factor=2)
        with self.assertRaises(ValueError):
            downsampler(torch.randn(1, 1, 3, 4))

    def test_custom_conv_block(self):
        custom_conv = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1)
        downsampler = SubpixelDownsample(spatial_dims=2, in_channels=1, scale_factor=2, conv_block=custom_conv)
        with eval_mode(downsampler):
            result = downsampler(torch.randn(1, 1, 4, 4))
            self.assertEqual(result.shape, (1, 8, 2, 2))


if __name__ == "__main__":
    unittest.main()
