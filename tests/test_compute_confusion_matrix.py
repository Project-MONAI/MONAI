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
from typing import Any, Dict, List

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import ConfusionMatrixMetric, get_confusion_matrix

# input data
data: Dict[Any, Any] = {
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
}

data_nan: Dict[Any, Any] = {
    # confusion matrix:[[[0,1,2,1],[1,1,1,1],[0,1,2,1]],
    #                   [[0,0,0,4],[0,0,4,0],[0,4,0,0]],
    #                   [[0,0,2,2],[0,0,2,2],[0,4,0,0]]]
    "y_pred": torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ]
    ),
    "y": torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ]
    ),
}

data_clf: Dict[Any, Any] = {
    "y_pred": torch.tensor([[1, 0, 0], [0, 0, 1]]),
    "y": torch.tensor([[1, 0, 0], [0, 1, 0]]),
}

# 1. test confusion matrix
TEST_CASE_CONFUSION_MATRIX = [
    data.copy(),
    torch.tensor(
        [
            [[0.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0]],
            [[1.0, 0.0, 3.0, 0.0], [1.0, 0.0, 2.0, 1.0], [1.0, 1.0, 2.0, 0.0]],
        ]
    ),
]

# 2. test metric with compute_sample
TEST_CASES_COMPUTE_SAMPLE = []
result_with_class = [
    torch.tensor([0.5000, 0.5000, 0.5000]),
    torch.tensor([0.1667, 0.2500, 0.3333]),
    torch.tensor([0.8333, 0.7500, 0.6667]),
    torch.tensor([0.5000, 0.7500, 0.2500]),
    torch.tensor([0.8333, 0.5833, 0.8333]),
    torch.tensor([0.5000, 0.5000, 0.5000]),
    torch.tensor([0.5000, 0.2500, 0.7500]),
    torch.tensor([0.1667, 0.4167, 0.1667]),
    torch.tensor([0.5000, 0.0000, 0.6830]),
    torch.tensor([0.5000, 0.4167, 0.2500]),
    torch.tensor([0.7500, 0.6250, 0.6250]),
    torch.tensor([0.6667, 0.6250, 0.5833]),
    torch.tensor([0.5000, 0.5833, 0.3333]),
    torch.tensor([0.3333, 0.2887, 0.1220]),
    torch.tensor([0.5000, 0.6036, 0.3536]),
    torch.tensor([0.3333, 0.2500, 0.1667]),
    torch.tensor([0.3333, 0.3333, 0.0833]),
]
result_without_class = [
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.7500]),
    torch.tensor([0.5000]),
    torch.tensor([0.7500]),
    torch.tensor([0.5000]),
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.5610]),
    torch.tensor([0.3889]),
    torch.tensor([0.6667]),
    torch.tensor([0.6250]),
    torch.tensor([0.4722]),
    torch.tensor([0.2480]),
    torch.tensor([0.4857]),
    torch.tensor([0.2500]),
    torch.tensor([0.2500]),
]
metric_names = [
    "tpr",
    "fpr",
    "tnr",
    "ppv",
    "npv",
    "fnr",
    "fdr",
    "for",
    "pt",
    "ts",
    "acc",
    "ba",
    "f1",
    "mcc",
    "fm",
    "bm",
    "mk",
]
for idx in range(len(metric_names)):
    for output_class in [True, False]:
        TEST_CASE: List[Any] = [data.copy()]
        TEST_CASE[0]["compute_sample"] = True
        TEST_CASE[0]["include_background"] = True
        TEST_CASE[0]["metric_name"] = metric_names[idx]
        TEST_CASE[0]["output_class"] = output_class
        if not output_class:
            result = result_without_class[idx]
        else:
            result = result_with_class[idx]
        TEST_CASE.append(result)
        TEST_CASES_COMPUTE_SAMPLE.append(TEST_CASE)

# 3. test metric with compute_sample, denominator may have zeros
TEST_CASES_COMPUTE_SAMPLE_NAN = []
metric_names = ["tpr", "tnr"]
result_with_class = [
    torch.tensor([0.0000, 0.2500, 0.0000]),
    torch.tensor([0.8333, 0.8333, 0.2222]),
]
not_nans_class = [
    torch.tensor([3.0, 2.0, 1.0]),
    torch.tensor([2.0, 3.0, 3.0]),
]
for idx in range(2):
    TEST_CASE = [data_nan.copy()]
    TEST_CASE[0]["compute_sample"] = True
    TEST_CASE[0]["include_background"] = True
    TEST_CASE[0]["output_class"] = True
    TEST_CASE[0]["metric_name"] = metric_names[idx]
    TEST_CASE.append(result_with_class[idx])
    TEST_CASE.append(not_nans_class[idx])
    TEST_CASES_COMPUTE_SAMPLE_NAN.append(TEST_CASE)

# 4. test metric without compute_sample
TEST_CASES_NO_COMPUTE_SAMPLE = []
result_with_class = [
    torch.tensor([0.5000, 0.5000, 0.5000]),
    torch.tensor([0.1667, 0.2500, 0.3333]),
    torch.tensor([0.8333, 0.7500, 0.6667]),
    torch.tensor([0.5000, 0.6667, 0.3333]),
    torch.tensor([0.8333, 0.6000, 0.8000]),
    torch.tensor([0.5000, 0.5000, 0.5000]),
    torch.tensor([0.5000, 0.3333, 0.6667]),
    torch.tensor([0.1667, 0.4000, 0.2000]),
    torch.tensor([0.3660, 0.4142, 0.4495]),
    torch.tensor([0.3333, 0.4000, 0.2500]),
    torch.tensor([0.7500, 0.6250, 0.6250]),
    torch.tensor([0.6667, 0.6250, 0.5833]),
    torch.tensor([0.5000, 0.5714, 0.4000]),
    torch.tensor([0.3333, 0.2582, 0.1491]),
    torch.tensor([0.5000, 0.5774, 0.4082]),
    torch.tensor([0.3333, 0.2500, 0.1667]),
    torch.tensor([0.3333, 0.2667, 0.1333]),
]
result_without_class = [
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.7500]),
    torch.tensor([0.5000]),
    torch.tensor([0.7500]),
    torch.tensor([0.5000]),
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.4142]),
    torch.tensor([0.3333]),
    torch.tensor([0.6667]),
    torch.tensor([0.6250]),
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.2500]),
]
metric_names = [
    "tpr",
    "fpr",
    "tnr",
    "ppv",
    "npv",
    "fnr",
    "fdr",
    "for",
    "pt",
    "ts",
    "acc",
    "ba",
    "f1",
    "mcc",
    "fm",
    "bm",
    "mk",
]
for idx in range(len(metric_names)):
    for output_class in [True, False]:
        TEST_CASE = [data.copy()]
        TEST_CASE[0]["compute_sample"] = False
        TEST_CASE[0]["include_background"] = True
        TEST_CASE[0]["metric_name"] = metric_names[idx]
        TEST_CASE[0]["output_class"] = output_class
        if not output_class:
            result = result_without_class[idx]
        else:
            result = result_with_class[idx]
        TEST_CASE.append(result)
        TEST_CASES_NO_COMPUTE_SAMPLE.append(TEST_CASE)

# 5. test classification task, denominator may have zeros
TEST_CASES_CLF_NAN = []
metric_names = ["tpr", "tnr"]
result_with_class = [
    torch.tensor([1.0000, 0.0000, 0.0000]),
    torch.tensor([1.0000, 1.0000, 0.5000]),
]
not_nans_class = [
    torch.tensor(2),
    torch.tensor(3),
]
for idx in range(2):
    TEST_CASE = [data_clf.copy()]
    TEST_CASE[0]["compute_sample"] = False
    TEST_CASE[0]["include_background"] = True
    TEST_CASE[0]["output_class"] = True
    TEST_CASE[0]["metric_name"] = metric_names[idx]
    TEST_CASE.append(result_with_class[idx])
    TEST_CASE.append(not_nans_class[idx])
    TEST_CASES_CLF_NAN.append(TEST_CASE)


class TestConfusionMatrix(unittest.TestCase):
    @parameterized.expand([TEST_CASE_CONFUSION_MATRIX])
    def test_value(self, input_data, expected_value):
        # include or ignore background
        input_data["include_background"] = True
        result = get_confusion_matrix(**input_data)
        np.testing.assert_allclose(result, expected_value, atol=1e-4)
        input_data["include_background"] = False
        result = get_confusion_matrix(**input_data)
        np.testing.assert_allclose(result, expected_value[:, 1:, :], atol=1e-4)

    @parameterized.expand(TEST_CASES_COMPUTE_SAMPLE)
    def test_compute_sample(self, input_data, expected_value):
        params = input_data.copy()
        vals = dict()
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        result, _ = metric(**vals)
        np.testing.assert_allclose(result, expected_value, atol=1e-4)

    @parameterized.expand(TEST_CASES_NO_COMPUTE_SAMPLE)
    def test_no_compute_sample(self, input_data, expected_value):
        params = input_data.copy()
        vals = dict()
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        result, _ = metric(**vals)
        np.testing.assert_allclose(result, expected_value, atol=1e-4)

    @parameterized.expand(TEST_CASES_COMPUTE_SAMPLE_NAN)
    def test_compute_sample_with_nan(self, input_data, expected_value, expected_not_nans):
        params = input_data.copy()
        vals = dict()
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        result, not_nans = metric(**vals)
        np.testing.assert_allclose(result, expected_value, atol=1e-4)
        np.testing.assert_allclose(not_nans, expected_not_nans, atol=1e-4)

    @parameterized.expand(TEST_CASES_CLF_NAN)
    def test_clf_with_nan(self, input_data, expected_value, expected_not_nans):
        params = input_data.copy()
        vals = dict()
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        result, not_nans = metric(**vals)
        np.testing.assert_allclose(result, expected_value, atol=1e-4)
        np.testing.assert_allclose(not_nans, expected_not_nans, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
