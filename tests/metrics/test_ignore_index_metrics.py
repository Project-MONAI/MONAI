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

import torch
from parameterized import parameterized

from monai.metrics import (
    ConfusionMatrixMetric,
    DiceMetric,
    GeneralizedDiceScore,
    HausdorffDistanceMetric,
    MeanIoU,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)
from monai.utils import optional_import

scipy, has_scipy = optional_import("scipy")

# Test cases for metrics with their specific required arguments
TEST_METRICS = [
    (DiceMetric, {"include_background": True, "reduction": "mean"}),
    (MeanIoU, {"include_background": True, "reduction": "mean"}),
    (GeneralizedDiceScore, {"include_background": True}),
    (ConfusionMatrixMetric, {"metric_name": "accuracy"}),
]

NO_BACKGROUND_METRICS = [
    (MeanIoU, {"include_background": False, "reduction": "mean"}),
    (ConfusionMatrixMetric, {"include_background": False, "metric_name": "accuracy"}),
]

# Metrics that require SciPy (Hausdorff and Surface metrics)
SCIPY_METRICS = [
    (HausdorffDistanceMetric, {"include_background": True}),
    (SurfaceDistanceMetric, {"include_background": True}),
    (SurfaceDiceMetric, {"class_thresholds": [0.5, 0.5], "include_background": True}),
]


@unittest.skipUnless(has_scipy, "Scipy required for surface metrics")
class TestIgnoreIndexMetrics(unittest.TestCase):
    @parameterized.expand(TEST_METRICS + SCIPY_METRICS)
    def test_metric_ignore_consistency(self, metric_class, kwargs):
        # Initialize metric with ignore_index
        metric = metric_class(ignore_index=255, **kwargs)

        # Batch size 1, 2 Classes, 4x4 Image
        # y_pred1 and y_pred2 differ ONLY in the bottom half (the ignore zone)
        y_pred1 = torch.zeros((1, 2, 4, 4))
        y_pred1[:, 1, 0:2, :] = 1.0  # Top half prediction

        y_pred2 = y_pred1.clone()
        y_pred2[:, 1, 2:4, :] = 1.0  # Bottom half prediction (different!)

        # Target: Top half is valid (0/1), Bottom half should be ignored
        # For ignore_index=255 (sentinel), we need to mark ignored pixels differently
        # Option 1: Use ignore_index as a class index (e.g., ignore_index=1)
        # Option 2: Keep one-hot but set ignored region to all zeros
        y = torch.zeros((1, 2, 4, 4))
        y[:, 1, 0:2, 0:2] = 1.0  # Top-left is class 1
        y[:, 0, 0:2, 2:4] = 1.0  # Top-right is class 0
        # Bottom half: leave as all zeros to indicate "no valid class"

        # Run metric for both predictions
        metric.reset()
        metric(y_pred=y_pred1, y=y)
        res1 = metric.aggregate()
        if isinstance(res1, list):
            res1 = res1[0]

        metric.reset()
        metric(y_pred=y_pred2, y=y)
        res2 = metric.aggregate()
        if isinstance(res2, list):
            res2 = res2[0]

        # The result must be identical because the spatial difference
        # is hidden by the ignore_index
        torch.testing.assert_close(res1, res2, msg=f"Failed for {metric_class.__name__}")

    @parameterized.expand(
        [(metric_class, kwargs, ignore_index) for metric_class, kwargs in TEST_METRICS + SCIPY_METRICS for ignore_index in (0, 1)]
    )
    def test_metric_ignore_class_index(self, metric_class, kwargs, ignore_index):
        metric = metric_class(ignore_index=ignore_index, **kwargs)

        ignored_rows = slice(0, 2) if ignore_index == 0 else slice(2, 4)
        ignored_channel = ignore_index

        y_pred1 = torch.zeros((1, 2, 4, 4))
        y_pred1[:, 0, 0:2, :] = 1.0
        y_pred1[:, 1, 2:4, :] = 1.0

        y_pred2 = y_pred1.clone()
        y_pred2[:, ignored_channel, ignored_rows, :] = 0.0

        y = torch.zeros((1, 2, 4, 4))
        y[:, 0, 0:2, :] = 1.0
        y[:, 1, 2:4, :] = 1.0

        metric.reset()
        metric(y_pred=y_pred1, y=y)
        res1 = metric.aggregate()
        if isinstance(res1, list):
            res1 = res1[0]

        metric.reset()
        metric(y_pred=y_pred2, y=y)
        res2 = metric.aggregate()
        if isinstance(res2, list):
            res2 = res2[0]

        torch.testing.assert_close(res1, res2, msg=f"Failed for {metric_class.__name__}")

    @parameterized.expand(NO_BACKGROUND_METRICS)
    def test_metric_ignore_class_index_without_background(self, metric_class, kwargs):
        metric = metric_class(ignore_index=1, **kwargs)

        y_pred1 = torch.zeros((1, 3, 4, 4))
        y_pred1[:, 1, 0:2, :] = 1.0
        y_pred1[:, 2, 2:4, :] = 1.0

        y_pred2 = y_pred1.clone()
        y_pred2[:, 1, 2:4, :] = 1.0

        y = torch.zeros((1, 3, 4, 4))
        y[:, 1, 0:2, :] = 1.0
        y[:, 2, 2:4, :] = 1.0

        metric.reset()
        metric(y_pred=y_pred1, y=y)
        res1 = metric.aggregate()
        if isinstance(res1, list):
            res1 = res1[0]

        metric.reset()
        metric(y_pred=y_pred2, y=y)
        res2 = metric.aggregate()
        if isinstance(res2, list):
            res2 = res2[0]

        torch.testing.assert_close(res1, res2, msg=f"Failed for {metric_class.__name__}")


if __name__ == "__main__":
    unittest.main()
