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
from monai.networks.nets import MedNeXt
from tests.utils import SkipIfBeforePyTorchVersion, test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_MEDNEXT = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for deep_supervision in [False, True]:
            for do_res in [False, True]:
                for do_res_up_down in [False, True]:
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "init_filters": init_filters,
                            "deep_supervision": deep_supervision,
                            "do_res": do_res,
                            "do_res_up_down": do_res_up_down,
                        },
                        (2, 1, *([16] * spatial_dims)),
                        (2, 2, *([16] * spatial_dims)),
                    ]
                    TEST_CASE_MEDNEXT.append(test_case)

TEST_CASE_MEDNEXT_2 = []
for spatial_dims in range(2, 4):
    for out_channels in [1, 2]:
        for deep_supervision in [False, True]:
            test_case = [
                {
                    "spatial_dims": spatial_dims,
                    "init_filters": 8,
                    "out_channels": out_channels,
                    "deep_supervision": deep_supervision,
                },
                (2, 1, *([16] * spatial_dims)),
                (2, out_channels, *([16] * spatial_dims)),
            ]
            TEST_CASE_MEDNEXT_2.append(test_case)


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


if __name__ == "__main__":
    unittest.main()
