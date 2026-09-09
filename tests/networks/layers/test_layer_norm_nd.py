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
import torch.nn as nn
from parameterized import parameterized

from monai.networks.layers import LayerNormNd


class TestLayerNormNd(unittest.TestCase):

    @parameterized.expand([[1], [2], [3]])
    def test_shape(self, spatial_dims):
        num_channels = 6
        norm = LayerNormNd(num_channels, spatial_dims=spatial_dims)
        x = torch.randn(2, num_channels, *([4] * spatial_dims))
        self.assertEqual(norm(x).shape, x.shape)

    @parameterized.expand([[2], [3]])
    def test_equivalent_to_layer_norm(self, spatial_dims):
        """`LayerNormNd` should equal `nn.LayerNorm` applied to the equivalent channels-last tensor."""
        num_channels = 6
        norm_nd = LayerNormNd(num_channels, spatial_dims=spatial_dims)
        norm_ref = nn.LayerNorm(num_channels, eps=1e-6)
        with torch.no_grad():
            norm_nd.weight.normal_()
            norm_nd.bias.normal_()
            norm_ref.weight.copy_(norm_nd.weight)
            norm_ref.bias.copy_(norm_nd.bias)

        x = torch.randn(2, num_channels, *([4] * spatial_dims))
        channel_last = list(range(spatial_dims + 2))
        channel_last.append(channel_last.pop(1))  # (batch, channel, *spatial) -> (batch, *spatial, channel)
        expected = norm_ref(x.permute(channel_last)).permute([0, spatial_dims + 1, *range(1, spatial_dims + 1)])
        self.assertTrue(torch.allclose(norm_nd(x), expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
