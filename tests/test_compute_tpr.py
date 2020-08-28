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
from typing import Tuple

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import compute_confusion_metric

metric_name = "true positive rate"
# test the metric under 2 class classification task
TEST_CASES_2_CLS_CLF = [
    [
        {
            "y_pred": torch.tensor([[0.4], [0.4], [0.4], [0.6], [0.9], [0.9]]),
            "y": torch.tensor([[1], [0], [0], [1], [1], [1]]),
            "bin_mode": "threshold",
            "metric_name": metric_name,
        },
        0.75,
    ],
    [
        {
            "y_pred": torch.tensor([[0], [0], [0], [1], [1], [1]]),
            "y": torch.tensor([[1], [0], [0], [1], [1], [1]]),
            "bin_mode": None,
            "metric_name": metric_name,
        },
        0.75,
    ],
]

# test the metric under multi-class classification task
average_list = ["micro", "macro", "weighted"]
multi_class_result_list = [0.4, 0.33333334, 0.4]
TEST_CASES_M_CLS_CLF = []
y_pred = torch.tensor([[0.4, 0.8, 1], [0.4, 0.8, 0], [0.4, 0.2, 0.7], [0.6, 0.1, 0.2], [0.2, 0.1, 0.8]])
y = torch.tensor([[1], [0], [2], [1], [2]])
for i in range(len(average_list)):
    average = average_list[i]
    test_case = [
        {
            "y_pred": y_pred,
            "y": y,
            "to_onehot_y": True,
            "metric_name": metric_name,
            "bin_mode": "mutually_exclusive",
            "average": average,
        },
        multi_class_result_list[i],
    ]
    TEST_CASES_M_CLS_CLF.append(test_case)


# test the metric under multi-label classification task
multi_label_result_list = [0.75, 0.80555556, 0.75]
TEST_CASES_M_LABEL_CLF = []
y_pred = torch.tensor([[0.4, 0.8, 1], [0.4, 0.8, 0], [0.4, 0.2, 0.7], [0.6, 0.1, 0.2], [0.2, 0.1, 0.8]])
y = torch.tensor([[0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1]])
for i in range(len(average_list)):
    average = average_list[i]
    test_case = [
        {
            "y_pred": y_pred,
            "y": y,
            "metric_name": metric_name,
            "bin_mode": "threshold",
            "bin_threshold": [0.3, 0.6, 0.5],
            "average": average,
        },
        multi_label_result_list[i],
    ]
    TEST_CASES_M_LABEL_CLF.append(test_case)


# test the metric under 2D segmentation task
def produce_seg_input(shape: Tuple):
    y = torch.cat([torch.ones(shape), torch.zeros(shape)])
    y_pred = torch.rand_like(y) * 0.5
    return y, y_pred


seg_result_list_2d = [0, 0, 0, 1, 1, 1]
basic_shape: Tuple
basic_shape = (5, 5, 3, 3)
y, y_org = produce_seg_input(basic_shape)
TEST_CASES_2D_SEG = []
ct = 0
for y_pred in [y_org, 1 - y_org]:
    for i in range(len(average_list)):
        average = average_list[i]
        test_case = [
            {"y_pred": y_pred, "y": y, "metric_name": metric_name, "bin_mode": "threshold", "average": average},
            seg_result_list_2d[ct],
        ]
        ct += 1
        TEST_CASES_2D_SEG.append(test_case)


# test the metric under 3D segmentation task
seg_result_list_3d = [1, 1, 1, 0, 0, 0]
basic_shape = (2, 5, 3, 3, 3)
y, y_org = produce_seg_input(basic_shape)
TEST_CASES_3D_SEG = []
ct = 0
for y_pred in [y_org, -y_org]:
    for i in range(len(average_list)):
        average = average_list[i]
        test_case = [
            {
                "y_pred": y_pred,
                "y": y,
                "activation": "sigmoid",
                "metric_name": metric_name,
                "bin_mode": "threshold",
                "average": average,
            },
            seg_result_list_3d[ct],
        ]
        ct += 1
        TEST_CASES_3D_SEG.append(test_case)


class TestComputeTprClf(unittest.TestCase):
    @parameterized.expand(TEST_CASES_2_CLS_CLF + TEST_CASES_M_CLS_CLF + TEST_CASES_M_LABEL_CLF)
    def test_value(self, input_data, expected_value):
        result = compute_confusion_metric(**input_data)
        np.testing.assert_allclose(expected_value, result, rtol=1e-7)


class TestComputeTprSeg(unittest.TestCase):
    @parameterized.expand(TEST_CASES_2D_SEG + TEST_CASES_3D_SEG)
    def test_value(self, input_data, expected_value):
        result = compute_confusion_metric(**input_data)
        self.assertEqual(expected_value, result)


if __name__ == "__main__":
    unittest.main()
