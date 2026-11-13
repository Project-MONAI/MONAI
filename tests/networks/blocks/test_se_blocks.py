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
from monai.networks.blocks import ChannelSELayer, ChannelSpatialSELayer, ResidualSELayer, SpatialSELayer
from tests.test_utils import test_script_save

TEST_CASES = [  # single channel 3D, batch 16
    [{"spatial_dims": 2, "in_channels": 4, "r": 3}, (7, 4, 64, 48), (7, 4, 64, 48)],  # 4-channel 2D, batch 7
    [  # 4-channel 1D, batch 16
        {"spatial_dims": 1, "in_channels": 4, "r": 3, "acti_type_1": "relu"},
        (16, 4, 63),
        (16, 4, 63),
    ],
]

TEST_CASES_3D = []
for type_1 in {"relu", "relu6", "leakyrelu"}:
    for type_2 in {"prelu", "sigmoid", "relu"}:
        test_case = [
            {"spatial_dims": 3, "in_channels": 10, "r": 3, "acti_type_1": type_1, "acti_type_2": type_2},
            (16, 10, 32, 24, 48),
            (16, 10, 32, 24, 48),
        ]
        TEST_CASES_3D.append(test_case)


class TestChannelSELayer(unittest.TestCase):
    @parameterized.expand(TEST_CASES + TEST_CASES_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = ChannelSELayer(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _ = TEST_CASES[0]
        net = ChannelSELayer(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            ChannelSELayer(spatial_dims=1, in_channels=4, r=100)


class TestResidualSELayer(unittest.TestCase):
    @parameterized.expand(TEST_CASES[:1])
    def test_shape(self, input_param, input_shape, expected_shape):
        net = ResidualSELayer(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _ = TEST_CASES[0]
        net = ResidualSELayer(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


TEST_CASES_SPATIAL = [
    [{"spatial_dims": 1, "in_channels": 4}, (16, 4, 63), (16, 4, 63)],
    [{"spatial_dims": 2, "in_channels": 4}, (7, 4, 64, 48), (7, 4, 64, 48)],
    [{"spatial_dims": 3, "in_channels": 10}, (2, 10, 32, 24, 48), (2, 10, 32, 24, 48)],
]


class TestSpatialSELayer(unittest.TestCase):
    """Test the spatial squeeze and channel excitation layer."""

    @parameterized.expand(TEST_CASES_SPATIAL)
    def test_shape(self, input_param, input_shape, expected_shape):
        """Verify the output keeps the input shape.

        Args:
            input_param: constructor arguments for the layer.
            input_shape: shape of the random input tensor.
            expected_shape: expected shape of the output.
        """
        net = SpatialSELayer(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_value(self):
        """The input is scaled by the sigmoid of a single-channel 1x1 convolution."""
        net = SpatialSELayer(spatial_dims=2, in_channels=3)
        x = torch.randn(2, 3, 8, 8)
        with eval_mode(net):
            result = net(x)
        expected = x * torch.sigmoid(net.conv(x))
        self.assertEqual(net.conv.out_channels, 1)
        self.assertTrue(torch.allclose(result, expected))

    def test_script(self):
        """Verify the layer can be saved with TorchScript."""
        input_param, input_shape, _ = TEST_CASES_SPATIAL[1]
        net = SpatialSELayer(**input_param)
        test_script_save(net, torch.randn(input_shape))


class TestChannelSpatialSELayer(unittest.TestCase):
    """Test the concurrent spatial and channel squeeze and excitation layer."""

    @parameterized.expand([[{**p, "r": 2}, i, o] for p, i, o in TEST_CASES_SPATIAL])
    def test_shape(self, input_param, input_shape, expected_shape):
        """Verify the output keeps the input shape.

        Args:
            input_param: constructor arguments for the layer.
            input_shape: shape of the random input tensor.
            expected_shape: expected shape of the output.
        """
        net = ChannelSpatialSELayer(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_value(self):
        """The output is the element-wise maximum of the channel and spatial SE outputs."""
        net = ChannelSpatialSELayer(spatial_dims=2, in_channels=4)
        x = torch.randn(2, 4, 8, 8)
        with eval_mode(net):
            result = net(x)
            expected = torch.max(net.cse(x), net.sse(x))
        self.assertTrue(torch.allclose(result, expected))

    def test_script(self):
        """Verify the layer can be saved with TorchScript."""
        input_param, input_shape, _ = TEST_CASES_SPATIAL[1]
        net = ChannelSpatialSELayer(**input_param)
        test_script_save(net, torch.randn(input_shape))

    def test_ill_arg(self):
        """Verify a reduction ratio larger than the channel count is rejected."""
        with self.assertRaises(ValueError):
            ChannelSpatialSELayer(spatial_dims=2, in_channels=4, r=100)


if __name__ == "__main__":
    unittest.main()
