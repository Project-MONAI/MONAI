# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from typing import Any, Sequence, Union

import torch
from parameterized import parameterized

from monai.networks.nets import DynUNet

# from tests.utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

strides: Sequence[Union[Sequence[int], int]]
kernel_size: Sequence[Any]
expected_shape: Sequence[Any]

TEST_CASE_DYNUNET_2D = []
for kernel_size in [(3, 3, 3, 1), ((3, 1), 1, (3, 3), (1, 1))]:
    for strides in [(1, 1, 1, 1), (2, 2, 2, 1)]:
        for in_channels in [2, 3]:
            for res_block in [True, False]:
                out_channels = 2
                in_size = 64
                spatial_dims = 2
                expected_shape = (1, out_channels, *[in_size // strides[0]] * spatial_dims)
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                        "strides": strides,
                        "upsample_kernel_size": strides[1:],
                        "norm_name": "batch",
                        "deep_supervision": False,
                        "res_block": res_block,
                    },
                    (1, in_channels, in_size, in_size),
                    expected_shape,
                ]
                TEST_CASE_DYNUNET_2D.append(test_case)

TEST_CASE_DYNUNET_3D = []  # in 3d cases, also test anisotropic kernel/strides
for out_channels in [2, 3]:
    for res_block in [True, False]:
        in_channels = 1
        in_size = 64
        expected_shape = (1, out_channels, 64, 32, 64)
        test_case = [
            {
                "spatial_dims": 3,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": (3, (1, 1, 3), 3, 3),
                "strides": ((1, 2, 1), 2, 2, 1),
                "upsample_kernel_size": (2, 2, 1),
                "norm_name": "instance",
                "deep_supervision": False,
                "res_block": res_block,
            },
            (1, in_channels, in_size, in_size, in_size),
            expected_shape,
        ]
        TEST_CASE_DYNUNET_3D.append(test_case)

TEST_CASE_DEEP_SUPERVISION = []
for spatial_dims in [2, 3]:
    for res_block in [True, False]:
        for deep_supr_num in [1, 2]:
            for strides in [(1, 2, 1, 2, 1), (2, 2, 2, 1), (2, 1, 1, 2, 2)]:
                test_case = [
                    {
                        "spatial_dims": spatial_dims,
                        "in_channels": 1,
                        "out_channels": 2,
                        "kernel_size": [3] * len(strides),
                        "strides": strides,
                        "upsample_kernel_size": strides[1:],
                        "norm_name": "group",
                        "deep_supervision": True,
                        "deep_supr_num": deep_supr_num,
                        "res_block": res_block,
                    },
                    (1, 1, *[in_size] * spatial_dims),
                ]
                scale = 1
                all_expected_shapes = []
                for stride in strides[: 1 + deep_supr_num]:
                    scale *= stride
                    deep_out_shape = (1, 2, *[in_size // scale] * spatial_dims)
                    all_expected_shapes.append(deep_out_shape)
                test_case.append(all_expected_shapes)
                TEST_CASE_DEEP_SUPERVISION.append(test_case)


class TestDynUNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_DYNUNET_2D + TEST_CASE_DYNUNET_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = DynUNet(**input_param).to(device)
        net.eval()
        with torch.no_grad():
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)


#     def test_script(self):
#         input_param, input_shape, _ = TEST_CASE_DYNUNET_2D[0]
#         net = DynUNet(**input_param)
#         test_data = torch.randn(input_shape)
#         test_script_save(net, test_data)


class TestDynUNetDeepSupervision(unittest.TestCase):
    @parameterized.expand(TEST_CASE_DEEP_SUPERVISION)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = DynUNet(**input_param).to(device)
        with torch.no_grad():
            results = net(torch.randn(input_shape).to(device))
            self.assertEqual(len(results), len(expected_shape))
            for idx in range(len(results)):
                result, sub_expected_shape = results[idx], expected_shape[idx]
                self.assertEqual(result.shape, sub_expected_shape)


if __name__ == "__main__":
    unittest.main()
