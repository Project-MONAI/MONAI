# Copyright 2020 - 2021 MONAI Consortium
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

from monai.transforms import Filter
from tests.utils import allclose, clone

grid_1 = torch.tensor(
    [
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        ]
    ]
)


TEST_CASE_0 = [
    "filter_single_label",
    {"applied_labels": 3},
    grid_1,
    torch.tensor(
        [
            [
                [
                    [0, 0, 3],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        ]
    ),
]


TEST_CASE_1 = [
    "filter_single_label_list",
    {"applied_labels": [3]},
    grid_1,
    torch.tensor(
        [
            [
                [
                    [0, 0, 3],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        ]
    ),
]

TEST_CASE_2 = [
    "filter_multi_label",
    {"applied_labels": [3, 5, 8]},
    grid_1,
    torch.tensor(
        [
            [
                [
                    [0, 0, 3],
                    [0, 5, 0],
                    [0, 8, 0],
                ]
            ]
        ]
    ),
]

TEST_CASE_3 = [
    "filter_all",
    {"applied_labels": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
    grid_1,
    grid_1,
]

VALID_CASES = [
    TEST_CASE_0,
    TEST_CASE_1,
    TEST_CASE_2,
    TEST_CASE_3,
]

ITEST_CASE_1 = ["invalid_image_data_type", {"applied_labels": 1}, [[[[1, 1, 1]]]], NotImplementedError]

INVALID_CASES = [ITEST_CASE_1]


class TestFillHoles(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, args, input_image, expected):
        converter = Filter(**args)
        if isinstance(input_image, torch.Tensor) and torch.cuda.is_available():
            result = converter(clone(input_image).cuda())
            assert allclose(result, expected.cuda())
        else:
            result = converter(clone(input_image))
            assert allclose(result, expected)

    @parameterized.expand(INVALID_CASES)
    def test_raise_exception(self, _, args, input_image, expected_error):
        with self.assertRaises(expected_error):
            converter = Filter(**args)
            if isinstance(input_image, torch.Tensor) and torch.cuda.is_available():
                _ = converter(clone(input_image).cuda())
            else:
                _ = converter(clone(input_image))


if __name__ == "__main__":
    unittest.main()
