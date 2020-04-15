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
from monai.transforms.composables_tensor import Flipd

TEST_CASE_1 = [
    {
        'keys': ['img'],
        'spatial_axis': 0
    },
    {'img': torch.tensor([
        [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
    ])},
    torch.tensor([
        [
            [5, 6],
            [3, 4],
            [1, 2]
        ]
    ])
]

TEST_CASE_2 = [
    {
        'keys': ['img'],
        'spatial_axis': [0, 1]
    },
    {'img': torch.tensor([
        [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
    ])},
    torch.tensor([
        [
            [6, 5],
            [4, 3],
            [2, 1]
        ]
    ])
]


class TestTensorFlipd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_value(self, input_param, input_data, expected_value):
        result = Flipd(**input_param)(input_data)
        torch.testing.assert_allclose(result['img'], expected_value)


if __name__ == '__main__':
    unittest.main()
