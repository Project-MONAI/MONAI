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

from monai.transforms.utils_morphological_ops import dilate, erode, get_morphological_filter_result_t
from tests.test_utils import TEST_NDARRAYS, assert_allclose

TESTS_SHAPE = []
for p in TEST_NDARRAYS:
    mask = torch.zeros(1, 1, 5, 5, 5)
    filter_size = 3
    TESTS_SHAPE.append([{"mask": p(mask), "filter_size": filter_size}, [1, 1, 5, 5, 5]])
    mask = torch.zeros(3, 2, 5, 5, 5)
    filter_size = 5
    TESTS_SHAPE.append([{"mask": p(mask), "filter_size": filter_size}, [3, 2, 5, 5, 5]])
    mask = torch.zeros(1, 1, 1, 1, 1)
    filter_size = 5
    TESTS_SHAPE.append([{"mask": p(mask), "filter_size": filter_size}, [1, 1, 1, 1, 1]])
    mask = torch.zeros(1, 1, 1, 1)
    filter_size = 5
    TESTS_SHAPE.append([{"mask": p(mask), "filter_size": filter_size}, [1, 1, 1, 1]])

TESTS_VALUE_T = []
filter_size = 3
mask = torch.ones(3, 2, 3, 3, 3)
TESTS_VALUE_T.append([{"mask": mask, "filter_size": filter_size, "pad_value": 1.0}, torch.ones(3, 2, 3, 3, 3)])
mask = torch.zeros(3, 2, 3, 3, 3)
TESTS_VALUE_T.append([{"mask": mask, "filter_size": filter_size, "pad_value": 0.0}, torch.zeros(3, 2, 3, 3, 3)])
mask = torch.ones(3, 2, 3, 3)
TESTS_VALUE_T.append([{"mask": mask, "filter_size": filter_size, "pad_value": 1.0}, torch.ones(3, 2, 3, 3)])
mask = torch.zeros(3, 2, 3, 3)
TESTS_VALUE_T.append([{"mask": mask, "filter_size": filter_size, "pad_value": 0.0}, torch.zeros(3, 2, 3, 3)])

TESTS_VALUE = []
for p in TEST_NDARRAYS:
    mask = torch.zeros(3, 2, 5, 5, 5)
    filter_size = 3
    TESTS_VALUE.append(
        [{"mask": p(mask), "filter_size": filter_size}, p(torch.zeros(3, 2, 5, 5, 5)), p(torch.zeros(3, 2, 5, 5, 5))]
    )
    mask = torch.ones(1, 1, 3, 3, 3)
    filter_size = 3
    TESTS_VALUE.append(
        [{"mask": p(mask), "filter_size": filter_size}, p(torch.ones(1, 1, 3, 3, 3)), p(torch.ones(1, 1, 3, 3, 3))]
    )
    mask = torch.ones(1, 2, 3, 3, 3)
    filter_size = 3
    TESTS_VALUE.append(
        [{"mask": p(mask), "filter_size": filter_size}, p(torch.ones(1, 2, 3, 3, 3)), p(torch.ones(1, 2, 3, 3, 3))]
    )
    mask = torch.zeros(3, 2, 3, 3, 3)
    mask[:, :, 1, 1, 1] = 1.0
    filter_size = 3
    TESTS_VALUE.append(
        [{"mask": p(mask), "filter_size": filter_size}, p(torch.zeros(3, 2, 3, 3, 3)), p(torch.ones(3, 2, 3, 3, 3))]
    )
    mask = torch.zeros(3, 2, 3, 3)
    mask[:, :, 1, 1] = 1.0
    filter_size = 3
    TESTS_VALUE.append(
        [{"mask": p(mask), "filter_size": filter_size}, p(torch.zeros(3, 2, 3, 3)), p(torch.ones(3, 2, 3, 3))]
    )


class TestMorph(unittest.TestCase):
    @parameterized.expand(TESTS_SHAPE)
    def test_shape(self, input_data, expected_result):
        result1 = erode(input_data["mask"], input_data["filter_size"])
        assert_allclose(result1.shape, expected_result, type_test=False, device_test=False, atol=0.0)

    @parameterized.expand(TESTS_VALUE_T)
    def test_value_t(self, input_data, expected_result):
        result1 = get_morphological_filter_result_t(
            input_data["mask"], input_data["filter_size"], input_data["pad_value"]
        )
        assert_allclose(result1, expected_result, type_test=False, device_test=False, atol=0.0)

    @parameterized.expand(TESTS_VALUE)
    def test_value(self, input_data, expected_erode_result, expected_dilate_result):
        result1 = erode(input_data["mask"], input_data["filter_size"])
        assert_allclose(result1, expected_erode_result, type_test=True, device_test=True, atol=0.0)
        result2 = dilate(input_data["mask"], input_data["filter_size"])
        assert_allclose(result2, expected_dilate_result, type_test=True, device_test=True, atol=0.0)


if __name__ == "__main__":
    unittest.main()
