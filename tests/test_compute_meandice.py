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

from monai.utils.compute_meandice import compute_meandice

# keep background
TEST_CASE_1 = [
    {
        'y_pred': torch.tensor([[[[1., -1.], [-1., 1.]]]]),
        'y': torch.tensor([[[[1., 0.], [1., 1.]]]]),
        'remove_bg': False,
        'is_onehot_targets': True,
        'logit_thresh': 0.5,
        'add_sigmoid': True,
        'add_softmax': False
    },
    0.8000
]

# remove background and not One-Hot target
TEST_CASE_2 = [
    {
        'y_pred': torch.tensor([[[[-1., 3.], [2., -4.]], [[0., -1.], [3., 2.]], [[0., 1.], [2., -1.]]],
                               [[[-2., 0.], [3., 1.]], [[0., 2.], [1., -2.]], [[-1., 2.], [4., 0.]]]]),
        'y': torch.tensor([[[[1, 2], [1, 0]]], [[[1, 1], [2, 0]]]]),
        'remove_bg': True,
        'is_onehot_targets': False,
        'logit_thresh': None,
        'add_sigmoid': False,
        'add_softmax': True
    },
    0.4583
]


class TestComputeMeanDice(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_value(self, input_data, expected_value):
        result = compute_meandice(**input_data)
        self.assertAlmostEqual(result.item(), expected_value, places=4)


if __name__ == '__main__':
    unittest.main()
