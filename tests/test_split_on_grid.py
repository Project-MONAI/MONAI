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

from monai.apps.pathology.transforms import SplitOnGrid
from tests.utils import TEST_NDARRAYS, assert_allclose

A11 = torch.randn(3, 2, 2)
A12 = torch.randn(3, 2, 2)
A21 = torch.randn(3, 2, 2)
A22 = torch.randn(3, 2, 2)

A1 = torch.cat([A11, A12], 2)
A2 = torch.cat([A21, A22], 2)
A = torch.cat([A1, A2], 1)

TEST_CASE_0 = [{"grid_size": (2, 2)}, A, torch.stack([A11, A12, A21, A22])]
TEST_CASE_1 = [{"grid_size": (2, 1)}, A, torch.stack([A1, A2])]
TEST_CASE_2 = [{"grid_size": (1, 2)}, A1, torch.stack([A11, A12])]
TEST_CASE_3 = [{"grid_size": (1, 2)}, A2, torch.stack([A21, A22])]
TEST_CASE_4 = [{"grid_size": (1, 1), "patch_size": (2, 2)}, A, torch.stack([A11])]
TEST_CASE_5 = [{"grid_size": 1, "patch_size": 4}, A, torch.stack([A])]
TEST_CASE_6 = [{"grid_size": 2, "patch_size": 2}, A, torch.stack([A11, A12, A21, A22])]
TEST_CASE_7 = [{"grid_size": 1}, A, torch.stack([A])]
TEST_CASE_8 = [
    {"grid_size": (2, 2), "patch_size": 2},
    torch.arange(12).reshape(1, 3, 4).to(torch.float32),
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

TEST_CASE_MC_0 = [{"grid_size": (2, 2)}, [A, A], [torch.stack([A11, A12, A21, A22]), torch.stack([A11, A12, A21, A22])]]
TEST_CASE_MC_1 = [{"grid_size": (2, 1)}, [A] * 5, [torch.stack([A1, A2])] * 5]
TEST_CASE_MC_2 = [{"grid_size": (1, 2)}, [A1, A2], [torch.stack([A11, A12]), torch.stack([A21, A22])]]

TEST_MULTIPLE = []
for p in TEST_NDARRAYS:
    TEST_MULTIPLE.append([p, *TEST_CASE_MC_0])
    TEST_MULTIPLE.append([p, *TEST_CASE_MC_1])
    TEST_MULTIPLE.append([p, *TEST_CASE_MC_2])


class TestSplitOnGrid(unittest.TestCase):
    @parameterized.expand(TEST_SINGLE)
    def test_split_patch_single_call(self, in_type, input_parameters, image, expected):
        input_image = in_type(image)
        splitter = SplitOnGrid(**input_parameters)
        output = splitter(input_image)
        assert_allclose(output, expected, type_test=False)

    @parameterized.expand(TEST_MULTIPLE)
    def test_split_patch_multiple_call(self, in_type, input_parameters, img_list, expected_list):
        splitter = SplitOnGrid(**input_parameters)
        for image, expected in zip(img_list, expected_list):
            input_image = in_type(image)
            output = splitter(input_image)
            assert_allclose(output, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
