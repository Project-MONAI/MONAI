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

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import GeneralizedDiceScore, compute_generalized_dice

# keep background
TEST_CASE_1 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1)
    {
        "y_pred": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        "y": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
        "include_background": True,
    },
    [0.8],
]

# remove background
TEST_CASE_2 = [  # y (2, 1, 2, 2), y_pred (2, 3, 2, 2), expected out (2) (no background)
    {
        "y_pred": torch.tensor(
            [
                [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
        "include_background": False,
    },
    [0.1667, 0.6667],
]

# should return 0 for both cases
TEST_CASE_3 = [
    {
        "y_pred": torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
        "include_background": True,
    },
    [0.0, 0.0],
]

TEST_CASE_4 = [
    {"include_background": True, "reduction": "mean_batch"},
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            ]
        ),
    },
    [0.5455],
]

TEST_CASE_5 = [
    {"include_background": True, "reduction": "sum_batch"},
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
    },
    1.0455,
]

TEST_CASE_6 = [{"y": torch.ones((2, 2, 3, 3)), "y_pred": torch.ones((2, 2, 3, 3))}, [1.0000, 1.0000]]

TEST_CASE_7 = [{"y": torch.zeros((2, 2, 3, 3)), "y_pred": torch.ones((2, 2, 3, 3))}, [0.0000, 0.0000]]

TEST_CASE_8 = [{"y": torch.ones((2, 2, 3, 3)), "y_pred": torch.zeros((2, 2, 3, 3))}, [0.0000, 0.0000]]

TEST_CASE_9 = [{"y": torch.zeros((2, 2, 3, 3)), "y_pred": torch.zeros((2, 2, 3, 3))}, [1.0000, 1.0000]]


class TestComputeMeanDice(unittest.TestCase):
    # Functional part tests
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8, TEST_CASE_9])
    def test_value(self, input_data, expected_value):
        result = compute_generalized_dice(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

    # Functional part tests
    @parameterized.expand([TEST_CASE_3])
    def test_nans(self, input_data, expected_value):
        result = compute_generalized_dice(**input_data)
        self.assertTrue(np.allclose(np.isnan(result.cpu().numpy()), expected_value))

    # Samplewise tests
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_value_class(self, input_data, expected_value):

        # same test as for compute_meandice
        vals = {}
        vals["y_pred"] = input_data.pop("y_pred")
        vals["y"] = input_data.pop("y")
        generalized_dice_score = GeneralizedDiceScore(**input_data)
        generalized_dice_score(**vals)
        result = generalized_dice_score.aggregate(reduction="none")
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

    # Aggregation tests
    @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
    def test_nans_class(self, params, input_data, expected_value):

        generalized_dice_score = GeneralizedDiceScore(**params)
        generalized_dice_score(**input_data)
        result = generalized_dice_score.aggregate()
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
