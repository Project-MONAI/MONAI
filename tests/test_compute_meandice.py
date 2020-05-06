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

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import compute_meandice

# keep background
TEST_CASE_1 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1, 1)
    {
        "y_pred": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
        "y": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
        "include_background": True,
        "to_onehot_y": False,
        "mutually_exclusive": False,
        "logit_thresh": 0.5,
        "add_sigmoid": True,
    },
    [[0.8]],
]

# remove background and not One-Hot target
TEST_CASE_2 = [  # y (2, 1, 2, 2), y_pred (2, 3, 2, 2), expected out (2, 2) (no background)
    {
        "y_pred": torch.tensor(
            [
                [[[-1.0, 3.0], [2.0, -4.0]], [[0.0, -1.0], [3.0, 2.0]], [[0.0, 1.0], [2.0, -1.0]]],
                [[[-2.0, 0.0], [3.0, 1.0]], [[0.0, 2.0], [1.0, -2.0]], [[-1.0, 2.0], [4.0, 0.0]]],
            ]
        ),
        "y": torch.tensor([[[[1.0, 2.0], [1.0, 0.0]]], [[[1.0, 1.0], [2.0, 0.0]]]]),
        "include_background": False,
        "to_onehot_y": True,
        "mutually_exclusive": True,
    },
    [[0.5000, 0.0000], [0.6666, 0.6666]],
]

# should return Nan for all labels=0 case and skip for MeanDice
TEST_CASE_3 = [
    {
        "y_pred": torch.zeros(2, 3, 2, 2),
        "y": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 0.0], [0.0, 1.0]]]]),
        "include_background": True,
        "to_onehot_y": True,
        "mutually_exclusive": True,
    },
    [[False, True, True], [False, False, True]],
]


class TestComputeMeanDice(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_value(self, input_data, expected_value):
        result = compute_meandice(**input_data)
        self.assertTrue(np.allclose(result.cpu().numpy(), expected_value, atol=1e-4))

    @parameterized.expand([TEST_CASE_3])
    def test_nans(self, input_data, expected_value):
        result = compute_meandice(**input_data)
        self.assertTrue(np.allclose(np.isnan(result.cpu().numpy()), expected_value))


if __name__ == "__main__":
    unittest.main()
