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
from monai.networks.blocks import DownSample, MaxAvgPool, SubpixelDownsample, SubpixelUpsample
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

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

TEST_CASES_DOWNSAMPLE = [
    [{"spatial_dims": 2, "in_channels": 4, "mode": "conv"}, (1, 4, 16, 16), (1, 4, 8, 8)],
    [{"spatial_dims": 2, "in_channels": 4, "out_channels": 8, "mode": "convgroup"}, (1, 4, 16, 16), (1, 8, 8, 8)],
    [{"spatial_dims": 3, "in_channels": 2, "mode": "maxpool"}, (1, 2, 16, 16, 16), (1, 2, 8, 8, 8)],
    [{"spatial_dims": 2, "in_channels": 4, "mode": "avgpool"}, (1, 4, 16, 16), (1, 4, 8, 8)],
    [{"spatial_dims": 2, "in_channels": 1, "mode": "pixelunshuffle"}, (1, 1, 16, 16), (1, 4, 8, 8)],
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

    def test_reconstruction_2d(self):
        input_tensor = torch.randn(1, 1, 4, 4)
        down = SubpixelDownsample(spatial_dims=2, in_channels=1, scale_factor=2, conv_block=None)
        up = SubpixelUpsample(spatial_dims=2, in_channels=4, scale_factor=2, conv_block=None, apply_pad_pool=False)
        with eval_mode(down), eval_mode(up):
            downsampled = down(input_tensor)
            reconstructed = up(downsampled)
            self.assertTrue(torch.allclose(input_tensor, reconstructed, rtol=1e-5))

    def test_reconstruction_3d(self):
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


class TestDownSample(unittest.TestCase):
    @parameterized.expand(TEST_CASES_DOWNSAMPLE)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = DownSample(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_pre_post_conv(self):
        net = DownSample(
            spatial_dims=2,
            in_channels=4,
            out_channels=8,
            mode="maxpool",
            pre_conv="default",
            post_conv=torch.nn.Conv2d(8, 16, 1),
        )
        with eval_mode(net):
            result = net(torch.randn(1, 4, 16, 16))
            self.assertEqual(result.shape, (1, 16, 8, 8))

    def test_pixelunshuffle_equivalence(self):
        class DownSampleLocal(torch.nn.Module):
            def __init__(self, n_feat: int):
                super().__init__()
                self.conv = torch.nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
                self.pixelunshuffle = torch.nn.PixelUnshuffle(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                x = self.pixelunshuffle(x)
                return x

        n_feat = 2
        x = torch.randn(1, n_feat, 64, 64)

        fix_weight_conv = torch.nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)

        monai_down = DownSample(
            spatial_dims=2,
            in_channels=n_feat,
            out_channels=n_feat // 2,
            mode="pixelunshuffle",
            pre_conv=fix_weight_conv,
        )

        local_down = DownSampleLocal(n_feat)
        local_down.conv.weight.data = fix_weight_conv.weight.data.clone()

        with eval_mode(monai_down), eval_mode(local_down):
            out_monai = monai_down(x)
            out_local = local_down(x)

        self.assertTrue(torch.allclose(out_monai, out_local, rtol=1e-5))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            DownSample(spatial_dims=2, in_channels=4, mode="invalid")

    def test_missing_channels(self):
        with self.assertRaises(ValueError):
            DownSample(spatial_dims=2, mode="conv")


if __name__ == "__main__":
    unittest.main()
