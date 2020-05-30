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

import torch
from parameterized import parameterized

from monai.transforms import KeepLargestConnectedComponentd

grid_1 = {
    "img": torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]])
}
grid_2 = {
    "img": torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 2], [1, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]])
}

TEST_CASE_1 = [
    "value_1",
    {"keys": ["img"], "independent": False, "applied_values": [1]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_2 = [
    "value_2",
    {"keys": ["img"], "independent": False, "applied_values": [2]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_3 = [
    "independent_value_1_2",
    {"keys": ["img"], "independent": True, "applied_values": [1, 2]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_4 = [
    "dependent_value_1_2",
    {"keys": ["img"], "independent": False, "applied_values": [1, 2]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_5 = [
    "value_1",
    {"keys": ["img"], "independent": True, "applied_values": [1]},
    grid_2,
    torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]]),
]

TEST_CASE_6 = [
    "independent_value_1_2",
    {"keys": ["img"], "independent": True, "applied_values": [1, 2]},
    grid_2,
    torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]]),
]

TEST_CASE_7 = [
    "dependent_value_1_2",
    {"keys": ["img"], "independent": False, "applied_values": [1, 2]},
    grid_2,
    torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]]),
]

TEST_CASE_8 = [
    "value_1_connect_1",
    {"keys": ["img"], "independent": False, "applied_values": [1], "connectivity": 1},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_9 = [
    "independent_value_1_2_connect_1",
    {"keys": ["img"], "independent": True, "applied_values": [1, 2], "connectivity": 1},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_10 = [
    "dependent_value_1_2_connect_1",
    {"keys": ["img"], "independent": False, "applied_values": [1, 2], "connectivity": 1},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_11 = [
    "value_0_background_3",
    {"keys": ["img"], "independent": False, "applied_values": [0], "background": 3},
    grid_1,
    torch.tensor([[[[3, 3, 1, 3, 3], [3, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_12 = [
    "all_0_batch_2",
    {"keys": ["img"], "independent": False, "applied_values": [1], "background": 3},
    {
        "img": torch.tensor(
            [
                [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
                [[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]],
            ]
        )
    },
    torch.tensor(
        [
            [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
            [[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 3, 3, 3], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0]]],
        ]
    ),
]

TEST_CASE_13 = [
    "none_postfix",
    {"keys": ["img"], "output_postfix": None, "independent": False, "applied_values": [1]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]]),
]

VALID_CASES = [
    TEST_CASE_1,
    TEST_CASE_2,
    TEST_CASE_3,
    TEST_CASE_4,
    TEST_CASE_5,
    TEST_CASE_6,
    TEST_CASE_7,
    TEST_CASE_8,
    TEST_CASE_9,
    TEST_CASE_10,
    TEST_CASE_11,
    TEST_CASE_12,
]


class TestKeepLargestConnectedComponentd(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, args, input_dict, expected):
        converter = KeepLargestConnectedComponentd(**args)
        if torch.cuda.is_available():
            input_dict["img"] = input_dict["img"].cuda()
            result = converter(input_dict)
            torch.allclose(result["img_largestcc"], expected.cuda())
        else:
            result = converter(input_dict)
            torch.allclose(result["img_largestcc"], expected)

    @parameterized.expand([TEST_CASE_13])
    def test_none_postfix(self, _, args, input_dict, expected):
        converter = KeepLargestConnectedComponentd(**args)
        input_dict["img"] = input_dict["img"].cpu()
        result = converter(input_dict)
        torch.allclose(result["img"], expected)


if __name__ == "__main__":
    unittest.main()
