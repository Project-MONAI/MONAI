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

import unittest

import torch
from parameterized import parameterized

from monai.apps.pathology.transforms import SplitOnGridDict
from tests.utils import TEST_NDARRAYS, assert_allclose

A11 = torch.randn(3, 2, 2)
A12 = torch.randn(3, 2, 2)
A21 = torch.randn(3, 2, 2)
A22 = torch.randn(3, 2, 2)

A1 = torch.cat([A11, A12], 2)
A2 = torch.cat([A21, A22], 2)
A = torch.cat([A1, A2], 1)

TEST_CASE_0 = [{"keys": "image", "grid_size": (2, 2)}, {"image": A}, torch.stack([A11, A12, A21, A22])]
TEST_CASE_1 = [{"keys": "image", "grid_size": (2, 1)}, {"image": A}, torch.stack([A1, A2])]
TEST_CASE_2 = [{"keys": "image", "grid_size": (1, 2)}, {"image": A1}, torch.stack([A11, A12])]
TEST_CASE_3 = [{"keys": "image", "grid_size": (1, 2)}, {"image": A2}, torch.stack([A21, A22])]
TEST_CASE_4 = [{"keys": "image", "grid_size": (1, 1), "patch_size": (2, 2)}, {"image": A}, torch.stack([A11])]
TEST_CASE_5 = [{"keys": "image", "grid_size": 1, "patch_size": 4}, {"image": A}, torch.stack([A])]
TEST_CASE_6 = [{"keys": "image", "grid_size": 2, "patch_size": 2}, {"image": A}, torch.stack([A11, A12, A21, A22])]
TEST_CASE_7 = [{"keys": "image", "grid_size": 1}, {"image": A}, torch.stack([A])]
TEST_CASE_8 = [
    {"keys": "image", "grid_size": (2, 2), "patch_size": 2},
    {"image": torch.arange(12).reshape(1, 3, 4).to(torch.float32)},
    torch.Tensor([[[[0, 1], [4, 5]]], [[[2, 3], [6, 7]]], [[[4, 5], [8, 9]]], [[[6, 7], [10, 11]]]]).to(torch.float32),
]

TEST_SINGLE = []
for p in TEST_NDARRAYS:
    TEST_SINGLE.append([p, *TEST_CASE_0])
    TEST_SINGLE.append([p, *TEST_CASE_1])
    TEST_SINGLE.append([p, *TEST_CASE_2])
    TEST_SINGLE.append([p, *TEST_CASE_3])
    TEST_SINGLE.append([p, *TEST_CASE_4])
    TEST_SINGLE.append([p, *TEST_CASE_5])
    TEST_SINGLE.append([p, *TEST_CASE_6])
    TEST_SINGLE.append([p, *TEST_CASE_7])
    TEST_SINGLE.append([p, *TEST_CASE_8])

TEST_CASE_MC_0 = [
    {"keys": "image", "grid_size": (2, 2)},
    [{"image": A}, {"image": A}],
    [torch.stack([A11, A12, A21, A22]), torch.stack([A11, A12, A21, A22])],
]
TEST_CASE_MC_1 = [
    {"keys": "image", "grid_size": (2, 1)},
    [{"image": A}, {"image": A}, {"image": A}],
    [torch.stack([A1, A2])] * 3,
]
TEST_CASE_MC_2 = [
    {"keys": "image", "grid_size": (1, 2)},
    [{"image": A1}, {"image": A2}],
    [torch.stack([A11, A12]), torch.stack([A21, A22])],
]

TEST_MULTIPLE = []
for p in TEST_NDARRAYS:
    TEST_MULTIPLE.append([p, *TEST_CASE_MC_0])
    TEST_MULTIPLE.append([p, *TEST_CASE_MC_1])
    TEST_MULTIPLE.append([p, *TEST_CASE_MC_2])


class TestSplitOnGridDict(unittest.TestCase):
    @parameterized.expand(TEST_SINGLE)
    def test_split_patch_single_call(self, in_type, input_parameters, img_dict, expected):
        input_dict = {}
        for k, v in img_dict.items():
            input_dict[k] = in_type(v)
        splitter = SplitOnGridDict(**input_parameters)
        output = splitter(input_dict)[input_parameters["keys"]]
        assert_allclose(output, expected, type_test=False)

    @parameterized.expand(TEST_MULTIPLE)
    def test_split_patch_multiple_call(self, in_type, input_parameters, img_list, expected_list):
        splitter = SplitOnGridDict(**input_parameters)
        for img_dict, expected in zip(img_list, expected_list):
            input_dict = {}
            for k, v in img_dict.items():
                input_dict[k] = in_type(v)
            output = splitter(input_dict)[input_parameters["keys"]]
            assert_allclose(output, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
