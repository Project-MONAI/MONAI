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

from monai.metrics import ConfusionMatrixMetric, compute_confusion_matrix_metric

# keep background
TEST_CASE_1 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1, 1)
    {
        "y_pred": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
        "y": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
        "include_background": True,
        "to_onehot_y": False,
        "bin_mode": "threshold",
        "bin_threshold": 0.5,
        "activation": "sigmoid",
        "metric_name": "tpr",
    },
    [[0.6667]],
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
        "bin_mode": "mutually_exclusive",
        "metric_name": "tnr",
    },
    [[0.5000, 0.6667], [1.0000, 0.6667]],
]

# should return Nan for all labels=0 case and skip for MeanDice
TEST_CASE_3 = [
    {
        "y_pred": torch.zeros(2, 3, 2, 2),
        "y": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 0.0], [0.0, 1.0]]]]),
        "include_background": True,
        "to_onehot_y": True,
        "bin_mode": "mutually_exclusive",
        "metric_name": "fpr",
    },
    [[True, False, False], [False, False, False]],
]

TEST_CASE_4 = [
    {"include_background": True, "to_onehot_y": True, "reduction": "mean_batch", "metric_name": "fnr"},
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 1.0], [2.0, 0.0]]]]),
    },
    [0.1250, 0.5000, 0.0000],
]

TEST_CASE_5 = [
    {"include_background": True, "to_onehot_y": True, "reduction": "mean", "metric_name": "f1"},
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[1.0, 1.0], [2.0, 0.0]]]]),
    },
    0.4040,
]

TEST_CASE_6 = [
    {"include_background": True, "to_onehot_y": True, "reduction": "sum_batch", "metric_name": "acc"},
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
    },
    [3.0000, 0.0000, 0.0000],
]

TEST_CASE_7 = [
    {"include_background": True, "to_onehot_y": True, "reduction": "mean", "metric_name": "ppv"},
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
    },
    0.3333,
]

TEST_CASE_8 = [
    {"to_onehot_y": True, "include_background": False, "reduction": "sum_batch", "metric_name": "npv"},
    {
        "y_pred": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ]
        ),
        "y": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
    },
    [2.0000, 2.0000],
]

TEST_CASE_9 = [
    {
        "metric_name": "fm",
        "y": torch.from_numpy(np.ones((2, 2, 3, 3))),
        "y_pred": torch.from_numpy(np.ones((2, 2, 3, 3))),
    },
    [[1.0000, 1.0000], [1.0000, 1.0000]],
]

TEST_CASE_10 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1, 1)
    {
        "y_pred": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
        "y": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
        "include_background": True,
        "to_onehot_y": False,
        "bin_mode": "threshold",
        "bin_threshold": 0.0,
        "activation": torch.tanh,
        "metric_name": "bm",
    },
    [[0.6667]],
]


class TestComputeMeanDice(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_9, TEST_CASE_10])
    def test_value(self, input_data, expected_value):
        result = compute_confusion_matrix_metric(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

    @parameterized.expand([TEST_CASE_3])
    def test_nans(self, input_data, expected_value):
        result = compute_confusion_matrix_metric(**input_data)
        self.assertTrue(np.allclose(np.isnan(result.cpu().numpy()), expected_value))

    # ConfusionMatrixMetric class tests
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_value_class(self, input_data, expected_value):

        # same test as for compute_confusion_matrix_metric
        vals = dict()
        vals["y_pred"] = input_data.pop("y_pred")
        vals["y"] = input_data.pop("y")
        metric = ConfusionMatrixMetric(**input_data, reduction="none")
        result = metric(**vals)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

    @parameterized.expand([TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7, TEST_CASE_8])
    def test_nans_class(self, params, input_data, expected_value):

        metric = ConfusionMatrixMetric(**params)
        result = metric(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
