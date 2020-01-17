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

from monai.networks.losses.dice import DiceLoss

TEST_CASE_1 = [
    {
        'include_background': False,
    },
    {
        'pred': torch.tensor([[[[1., -1.], [-1., 1.]]]]),
        'ground': torch.tensor([[[[1., 0.], [1., 1.]]]]),
        'smooth': 1e-6,
    },
    0.307576,
]

TEST_CASE_2 = [
    {
        'include_background': True,
    },
    {
        'pred': torch.tensor([[[[1., -1.], [-1., 1.]]], [[[1., -1.], [-1., 1.]]]]),
        'ground': torch.tensor([[[[1., 1.], [1., 1.]]], [[[1., 0.], [1., 0.]]]]),
        'smooth': 1e-4,
    },
    0.416636,
]


class TestDiceLoss(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_val):
        result = DiceLoss(**input_param).forward(**input_data)
        self.assertAlmostEqual(result.item(), expected_val, places=5)


if __name__ == '__main__':
    unittest.main()
