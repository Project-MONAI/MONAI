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

from monai.metrics import (
    ConfusionMatrixMetric,
    DiceMetric,
    GeneralizedDiceScore,
    HausdorffDistanceMetric,
    MeanIoU,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)

# Test cases for metrics with their specific required arguments
TEST_METRICS = [
    (DiceMetric, {"include_background": True, "reduction": "mean"}),
    (MeanIoU, {"include_background": True, "reduction": "mean"}),
    (GeneralizedDiceScore, {"include_background": True}),
    (ConfusionMatrixMetric, {"metric_name": "accuracy"}),
]

# Metrics that require SciPy (Hausdorff and Surface metrics)
SCIPY_METRICS = [
    (HausdorffDistanceMetric, {"include_background": True}),
    (SurfaceDistanceMetric, {"include_background": True}),
    (SurfaceDiceMetric, {"class_thresholds": [0.5, 0.5], "include_background": True}),
]


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

        # Target: Top half is valid (0/1), Bottom half is 255
        y = torch.zeros((1, 2, 4, 4))
        y[:, 1, 0:2, 0:2] = 1.0
        y[:, :, 2:4, :] = 255

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


if __name__ == "__main__":
    unittest.main()
