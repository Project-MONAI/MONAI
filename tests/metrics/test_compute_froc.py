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

from __future__ import annotations

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import compute_fp_tp_probs, compute_fp_tp_probs_nd, compute_froc_curve_data, compute_froc_score

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_CASE_1 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8], device=_device),
        "y_coord": torch.tensor([0, 2, 3], device=_device),
        "x_coord": torch.tensor([3, 0, 1], device=_device),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "labels_to_exclude": [2],
        "resolution_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    2,
]

TEST_CASE_2 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "y_coord": torch.tensor([0, 2, 3]),
        "x_coord": torch.tensor([3, 0, 1]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "resolution_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_3 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "y_coord": torch.tensor([0, 4, 6]),
        "x_coord": torch.tensor([6, 0, 2]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "resolution_level": 1,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_EXCLUDE_ABSENT = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "y_coord": torch.tensor([0, 2, 3]),
        "x_coord": torch.tensor([3, 0, 1]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "labels_to_exclude": [5],
        "resolution_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_EXCLUDE_DUPLICATE = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "y_coord": torch.tensor([0, 2, 3]),
        "x_coord": torch.tensor([3, 0, 1]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "labels_to_exclude": [2, 2],
        "resolution_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    2,
]

TEST_CASE_4 = [
    {
        "fp_probs": np.array([0.8, 0.6]),
        "tp_probs": np.array([1, 1, 0, 0, 0.8, 0.8, 0]),
        "num_targets": 4,
        "num_images": 2,
    },
    (0.25, 0.5, 1, 2, 4, 8),
    0.95833333,
]

TEST_CASE_5 = [
    {
        "fp_probs": torch.tensor([0.8, 0.6]),
        "tp_probs": torch.tensor([1, 1, 0, 0, 0.8, 0.8, 0]),
        "num_targets": 4,
        "num_images": 2,
    },
    (0.25),
    0.75,
]

TEST_CASE_ND_1 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "coords": torch.tensor([[0, 3], [2, 0], [3, 1]]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_ND_2 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "coords": torch.tensor([[0, 0, 3], [1, 2, 0], [0, 3, 1]]),
        "evaluation_mask": np.array(
            [
                [[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]],
                [[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]],
            ]
        ),
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]


class TestComputeFpTp(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_EXCLUDE_ABSENT, TEST_CASE_EXCLUDE_DUPLICATE]
    )
    def test_value(self, input_data, expected_fp, expected_tp, expected_num):
        fp_probs, tp_probs, num_tumors = compute_fp_tp_probs(**input_data)
        np.testing.assert_allclose(fp_probs, expected_fp, rtol=1e-5)
        np.testing.assert_allclose(tp_probs, expected_tp, rtol=1e-5)
        np.testing.assert_equal(num_tumors, expected_num)


class TestComputeFpTpNd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_ND_1, TEST_CASE_ND_2])
    def test_value(self, input_data, expected_fp, expected_tp, expected_num):
        fp_probs, tp_probs, num_tumors = compute_fp_tp_probs_nd(**input_data)
        np.testing.assert_allclose(fp_probs, expected_fp, rtol=1e-5)
        np.testing.assert_allclose(tp_probs, expected_tp, rtol=1e-5)
        np.testing.assert_equal(num_tumors, expected_num)


class TestComputeFrocScore(unittest.TestCase):
    @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
    def test_value(self, input_data, thresholds, expected_score):
        fps_per_image, total_sensitivity = compute_froc_curve_data(**input_data)
        score = compute_froc_score(fps_per_image, total_sensitivity, thresholds)
        np.testing.assert_allclose(score, expected_score, rtol=1e-5)


def _reference_compute_froc_curve_data(fp_probs, tp_probs, num_targets, num_images):
    """Original threshold-by-threshold implementation, retained for regression comparisons."""
    total_fps, total_tps = [], []
    all_probs = sorted(set(list(fp_probs) + list(tp_probs)))
    for thresh in all_probs[1:]:
        total_fps.append((fp_probs >= thresh).sum())
        total_tps.append((tp_probs >= thresh).sum())
    total_fps.append(0)
    total_tps.append(0)
    return np.asarray(total_fps) / float(num_images), np.asarray(total_tps) / float(num_targets)


class TestComputeFrocCurveData(unittest.TestCase):
    @parameterized.expand(
        [
            ("duplicates", [0.2, 0.5, 0.5, 0.9], [0.1, 0.5, 0.5, 0.8]),
            ("mostly_unique", [0.12, 0.34, 0.56, 0.78], [0.23, 0.45, 0.67, 0.89]),
            ("all_equal", [0.4, 0.4], [0.4, 0.4, 0.4]),
            ("empty", [], []),
            ("empty_fp", [], [0.2, 0.6, 0.6]),
            ("empty_tp", [0.2, 0.6, 0.6], []),
            ("threshold_equality", [0.2, 0.5, 0.8], [0.5, 0.5, 0.9]),
        ]
    )
    def test_matches_reference(self, _, fp_probs, tp_probs):
        fp_probs = np.asarray(fp_probs)
        tp_probs = np.asarray(tp_probs)
        actual = compute_froc_curve_data(fp_probs, tp_probs, num_targets=3, num_images=2)
        expected = _reference_compute_froc_curve_data(fp_probs, tp_probs, num_targets=3, num_images=2)
        for actual_array, expected_array in zip(actual, expected):
            np.testing.assert_array_equal(actual_array, expected_array)
            self.assertEqual(actual_array.dtype, expected_array.dtype)

    def test_nan_fp_probs_are_not_counted(self):
        fp_probs = np.asarray([np.nan])
        tp_probs = np.asarray([0.5, 0.8])
        fps_per_image, _ = compute_froc_curve_data(fp_probs, tp_probs, num_targets=2, num_images=1)
        np.testing.assert_array_equal(fps_per_image, np.zeros_like(fps_per_image))


if __name__ == "__main__":
    unittest.main()
