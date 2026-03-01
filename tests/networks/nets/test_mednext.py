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
from monai.networks.nets import MedNeXt, MedNeXtL, MedNeXtM, MedNeXtS
from tests.test_utils import dict_product  # Import dict_product

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_MEDNEXT = [
    [params, (2, 1, *([16] * params["spatial_dims"])), (2, 2, *([16] * params["spatial_dims"]))]
    for params in dict_product(
        spatial_dims=range(2, 4),
        init_filters=[8, 16],
        deep_supervision=[False, True],
        use_residual_connection=[False, True],
    )
]
TEST_CASE_MEDNEXT_2 = [
    [params, (2, 1, *([16] * params["spatial_dims"])), (2, params["out_channels"], *([16] * params["spatial_dims"]))]
    for params in dict_product(
        spatial_dims=range(2, 4), out_channels=[1, 2], deep_supervision=[False, True], init_filters=[8]
    )
]


TEST_CASE_MEDNEXT_VARIANTS = [
    [
        params["model"],
        {"spatial_dims": params["spatial_dims"], "in_channels": 1, "out_channels": params["out_channels"]},
        (2, 1, *([16] * params["spatial_dims"])),
        (2, params["out_channels"], *([16] * params["spatial_dims"])),
    ]
    for params in dict_product(
        model=[MedNeXtS, MedNeXtM, MedNeXtL], spatial_dims=range(2, 4), out_channels=[1, 2], in_channels=[1]
    )
]


class TestMedNeXt(unittest.TestCase):
    @parameterized.expand(TEST_CASE_MEDNEXT)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = MedNeXt(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            if input_param["deep_supervision"] and net.training:
                assert isinstance(result, tuple)
                self.assertEqual(result[0].shape, expected_shape, msg=str(input_param))
            else:
                self.assertEqual(result.shape, expected_shape, msg=str(input_param))

    @parameterized.expand(TEST_CASE_MEDNEXT_2)
    def test_shape2(self, input_param, input_shape, expected_shape):
        net = MedNeXt(**input_param).to(device)

        net.train()
        result = net(torch.randn(input_shape).to(device))
        if input_param["deep_supervision"]:
            assert isinstance(result, tuple)
            self.assertEqual(result[0].shape, expected_shape, msg=str(input_param))
        else:
            assert isinstance(result, torch.Tensor)
            self.assertEqual(result.shape, expected_shape, msg=str(input_param))

        net.eval()
        result = net(torch.randn(input_shape).to(device))
        assert isinstance(result, torch.Tensor)
        self.assertEqual(result.shape, expected_shape, msg=str(input_param))

    def test_ill_arg(self):
        with self.assertRaises(AssertionError):
            MedNeXt(spatial_dims=4)

    @parameterized.expand(TEST_CASE_MEDNEXT_VARIANTS)
    def test_mednext_variants(self, model, input_param, input_shape, expected_shape):
        net = model(**input_param).to(device)

        net.train()
        result = net(torch.randn(input_shape).to(device))
        assert isinstance(result, torch.Tensor)
        self.assertEqual(result.shape, expected_shape, msg=str(input_param))

        net.eval()
        with torch.no_grad():
            result = net(torch.randn(input_shape).to(device))
        assert isinstance(result, torch.Tensor)
        self.assertEqual(result.shape, expected_shape, msg=str(input_param))


if __name__ == "__main__":
    unittest.main()
