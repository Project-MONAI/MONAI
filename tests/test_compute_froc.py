# Copyright 2020 - 2021 MONAI Consortium
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

from monai.metrics import compute_fp_tp_probs, compute_froc_score

TEST_CASE_1 = [
    {
        "Probs": torch.tensor([1, 0.6, 0.8]),
        "Ycorr": torch.tensor([0, 2, 3]),
        "Xcorr": torch.tensor([3, 0, 1]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "Isolated_Tumor_Cells": [2],
        "image_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    2,
]

TEST_CASE_2 = [
    {
        "Probs": torch.tensor([1, 0.6, 0.8]),
        "Ycorr": torch.tensor([0, 2, 3]),
        "Xcorr": torch.tensor([3, 0, 1]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "image_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_3 = [
    {
        "Probs": torch.tensor([1, 0.6, 0.8]),
        "Ycorr": torch.tensor([0, 4, 6]),
        "Xcorr": torch.tensor([6, 0, 2]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "image_level": 1,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_4 = [
    {
        "FP_probs": np.array([0.8, 0.6]),
        "TP_probs": np.array([1, 1, 0, 0, 0.8, 0.8, 0]),
        "num_of_tumors": 4,
        "num_of_samples": 2,
    },
    0.95833333,
]

TEST_CASE_5 = [
    {
        "FP_probs": torch.tensor([0.8, 0.6]),
        "TP_probs": torch.tensor([1, 1, 0, 0, 0.8, 0.8, 0]),
        "num_of_tumors": 4,
        "num_of_samples": 2,
        "eval_thresholds": [0.25],
    },
    0.75,
]


class TestComputeFpTp(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, input_data, expected_fp, expected_tp, expected_num):
        fp_probs, tp_probs, num_tumors = compute_fp_tp_probs(**input_data)
        np.testing.assert_allclose(fp_probs, expected_fp, rtol=1e-5)
        np.testing.assert_allclose(tp_probs, expected_tp, rtol=1e-5)
        np.testing.assert_equal(num_tumors, expected_num)


class TestComputeFrocScore(unittest.TestCase):
    @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
    def test_value(self, input_data, expected_score):
        avg_sensitivity = compute_froc_score(**input_data)
        np.testing.assert_allclose(avg_sensitivity, expected_score, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
