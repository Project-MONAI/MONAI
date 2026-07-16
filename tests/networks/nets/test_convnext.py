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

from monai.networks import eval_mode
from monai.networks.nets import ConvNeXt, ConvNeXtBase, ConvNeXtLarge, ConvNeXtSmall, ConvNeXtTiny, ConvNeXtXLarge
from monai.networks.nets.convnext import LayerNormNd
from tests.test_utils import skip_if_quick, test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

# the variants only differ in `depths` and `features`, `features` is overridden throughout to keep the
# tests small while still exercising each variant's real depth configuration.
SMALL_FEATURES = (4, 8, 16, 32)

TEST_CASE_1 = [  # 2-channel 3D, batch 2
    {"spatial_dims": 3, "in_channels": 2, "out_channels": 3, "features": SMALL_FEATURES},
    (2, 2, 32, 32, 32),
    (2, 3),
]
TEST_CASE_2 = [  # 2-channel 2D, batch 2, non-cubic input
    {"spatial_dims": 2, "in_channels": 2, "out_channels": 3, "features": SMALL_FEATURES, "act": "relu"},
    (2, 2, 32, 64),
    (2, 3),
]
TEST_CASE_3 = [  # 1-channel 1D, batch 1
    {"spatial_dims": 1, "in_channels": 1, "out_channels": 3, "features": SMALL_FEATURES},
    (1, 1, 64),
    (1, 3),
]
TEST_CASE_4 = [  # stochastic depth and no layer scale
    {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "features": SMALL_FEATURES,
        "drop_path_rate": 0.2,
        "layer_scale_init_value": 0.0,
        "kernel_size": 3,
    },
    (2, 1, 32, 32, 32),
    (2, 2),
]

CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4]

TEST_CASES = []
for case in CASES:
    TEST_CASES.append([ConvNeXt, *case])

# every variant, over the 2D and 3D cases, exercising the aliases alongside the canonical names
TEST_VARIANT_CASES = []
for model in [ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge]:
    for case in [TEST_CASE_1, TEST_CASE_2]:
        TEST_VARIANT_CASES.append([model, *case])

TEST_SCRIPT_CASES = [[ConvNeXt, *case] for case in CASES]

# published ImageNet-1k parameter counts of the official 2D implementation,
# https://github.com/facebookresearch/ConvNeXt
TEST_REFERENCE_PARAMS = [[ConvNeXtTiny, 28_589_128], [ConvNeXtSmall, 50_223_688], [ConvNeXtBase, 88_591_464]]


class TestConvNeXt(unittest.TestCase):

    @parameterized.expand(TEST_CASES + TEST_VARIANT_CASES)
    def test_convnext_shape(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_SCRIPT_CASES)
    def test_script(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)

    @parameterized.expand(TEST_REFERENCE_PARAMS)
    @skip_if_quick
    def test_reference_parameter_count(self, model, expected_params):
        """The default variants should match the published parameter counts of the 2D reference models."""
        net = model(spatial_dims=2, in_channels=3, out_channels=1000)
        self.assertEqual(sum(p.numel() for p in net.parameters()), expected_params)

    def test_drop_path_schedule(self):
        """Stochastic depth should increase linearly from 0 over the blocks of the whole network."""
        net = ConvNeXt(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            depths=(1, 1, 2, 1),
            features=SMALL_FEATURES,
            drop_path_rate=0.3,
        )
        rates = [
            getattr(b.drop_path, "drop_prob", 0.0) for i in range(1, 5) for b in getattr(net.features, f"stage{i}")
        ]
        self.assertEqual(rates, sorted(rates))
        self.assertEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[-1], 0.3)
        # a zero rate should not add a drop path module at all
        self.assertIsInstance(net.features.stage1[0].drop_path, nn.Identity)

    def test_layer_scale(self):
        """`layer_scale_init_value` should control whether the residual branch is scaled."""
        net = ConvNeXt(spatial_dims=2, in_channels=1, out_channels=2, features=SMALL_FEATURES)
        self.assertIsNotNone(net.features.stage1[0].gamma)
        net = ConvNeXt(
            spatial_dims=2, in_channels=1, out_channels=2, features=SMALL_FEATURES, layer_scale_init_value=0.0
        )
        self.assertIsNone(net.features.stage1[0].gamma)

    @parameterized.expand([[2], [3]])
    def test_layer_norm_nd(self, spatial_dims):
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
        assert torch.allclose(norm_nd(x), expected, atol=1e-5)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):  # unsupported spatial_dims
            ConvNeXt(spatial_dims=4, in_channels=1, out_channels=2)
        with self.assertRaises(ValueError):  # depths and features of different lengths
            ConvNeXt(spatial_dims=3, in_channels=1, out_channels=2, depths=(1, 1), features=SMALL_FEATURES)


if __name__ == "__main__":
    unittest.main()
