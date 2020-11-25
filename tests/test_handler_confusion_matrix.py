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
from typing import Any, Dict

import torch
from parameterized import parameterized

from monai.handlers import ConfusionMatrix

TEST_CASE_1 = [{"include_background": True, "metric_name": "f1", "compute_sample": False}, 0.75]
TEST_CASE_2 = [{"include_background": False, "metric_name": "ppv", "compute_sample": False}, 1.0]

TEST_CASE_SEG_1 = [{"include_background": True, "metric_name": "tpr", "compute_sample": True}, 0.8333]
TEST_CASE_SEG_2 = [{"include_background": True, "metric_name": "tpr", "compute_sample": False}, 0.7]

data_1: Dict[Any, Any] = {
    "y_pred": torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
        ]
    ),
    "y": torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
        ]
    ),
}

data_2: Dict[Any, Any] = {
    "y_pred": torch.tensor(
        [
            [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]],
        ]
    ),
    "y": torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ]
    ),
}


class TestHandlerConfusionMatrix(unittest.TestCase):
    # TODO test multi node averaged confusion matrix

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_compute(self, input_params, expected_avg):
        metric = ConfusionMatrix(**input_params)

        y_pred = torch.Tensor([[[0], [1]], [[1], [0]]])
        y = torch.Tensor([[[0], [1]], [[0], [1]]])
        metric.update([y_pred, y])

        y_pred = torch.Tensor([[[0], [1]], [[1], [0]]])
        y = torch.Tensor([[[0], [1]], [[1], [0]]])
        metric.update([y_pred, y])

        avg_metric = metric.compute()
        self.assertAlmostEqual(avg_metric, expected_avg, places=4)

    @parameterized.expand([TEST_CASE_SEG_1, TEST_CASE_SEG_2])
    def test_compute_seg(self, input_params, expected_avg):
        metric = ConfusionMatrix(**input_params)

        y_pred = data_1["y_pred"]
        y = data_1["y"]
        metric.update([y_pred, y])

        y_pred = data_2["y_pred"]
        y = data_2["y"]
        metric.update([y_pred, y])

        avg_metric = metric.compute()
        if input_params["compute_sample"] is False:
            avg_metric = avg_metric.item()
        self.assertAlmostEqual(avg_metric, expected_avg, places=4)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape_mismatch(self, input_params, _expected):
        metric = ConfusionMatrix(**input_params)
        with self.assertRaises((AssertionError, ValueError)):
            y_pred = torch.Tensor([[0, 1], [1, 0]])
            y = torch.ones((2, 3))
            metric.update([y_pred, y])

        with self.assertRaises((AssertionError, ValueError)):
            y_pred = torch.Tensor([[0, 1], [1, 0]])
            y = torch.ones((3, 2))
            metric.update([y_pred, y])


if __name__ == "__main__":
    unittest.main()
