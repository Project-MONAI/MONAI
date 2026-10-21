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

from monai.metrics import AbsoluteVolumeDifferenceMetric, compute_absolute_volume_difference


class TestComputeAbsoluteVolumeDifference(unittest.TestCase):
    """Tests for the standalone compute_absolute_volume_difference function."""

    def test_perfect_prediction_returns_zero(self):
        """Identical prediction and ground truth should yield AVD of zero for all classes."""
        # identical masks → AVD = 0 for every class
        y = torch.zeros(2, 3, 4, 4)
        y[:, 1, :2, :2] = 1.0
        y[:, 2, 2:, 2:] = 1.0
        result = compute_absolute_volume_difference(y_pred=y, y=y, ignore_empty=False)
        self.assertEqual(result.shape, torch.Size([2, 3]))
        self.assertTrue(torch.all(result == 0.0))

    def test_known_volume_difference(self):
        """AVD should equal the absolute difference in foreground voxel counts between prediction and GT."""
        # batch=1, 2 classes (background + foreground), 1D spatial of length 10
        y_pred = torch.zeros(1, 2, 10)
        y_true = torch.zeros(1, 2, 10)
        y_pred[0, 1, :7] = 1.0  # 7 foreground voxels predicted
        y_true[0, 1, :4] = 1.0  # 4 foreground voxels in GT
        result = compute_absolute_volume_difference(y_pred=y_pred, y=y_true, ignore_empty=False)
        # channel 0: both all-zeros → AVD = 0
        # channel 1: |7 - 4| = 3
        self.assertAlmostEqual(result[0, 0].item(), 0.0)
        self.assertAlmostEqual(result[0, 1].item(), 3.0)

    def test_ignore_background(self):
        """Setting include_background=False should strip the first channel and reduce output shape accordingly."""
        y_pred = torch.zeros(2, 3, 8, 8)
        y_true = torch.zeros(2, 3, 8, 8)
        y_pred[:, 1, :3, :3] = 1.0
        y_true[:, 1, :4, :4] = 1.0
        result = compute_absolute_volume_difference(y_pred=y_pred, y=y_true, include_background=False)
        # background channel stripped → shape [2, 2]
        self.assertEqual(result.shape, torch.Size([2, 2]))

    def test_ignore_empty_sets_nan(self):
        """Channels with no ground-truth foreground voxels should be NaN when ignore_empty=True."""
        # channel 1 has no GT voxels → should be NaN when ignore_empty=True
        y_pred = torch.zeros(1, 2, 6)
        y_true = torch.zeros(1, 2, 6)
        y_pred[0, 0, :3] = 1.0
        result = compute_absolute_volume_difference(y_pred=y_pred, y=y_true, ignore_empty=True)
        # channel 0: GT is empty → NaN
        self.assertTrue(torch.isnan(result[0, 0]))
        # channel 1: GT is empty → NaN
        self.assertTrue(torch.isnan(result[0, 1]))

    def test_ignore_empty_false_returns_pred_volume(self):
        """With ignore_empty=False and empty GT, AVD should equal the predicted volume."""
        # when GT is all zero and ignore_empty=False, AVD = |V_pred - 0| = V_pred
        y_pred = torch.zeros(1, 2, 6)
        y_true = torch.zeros(1, 2, 6)
        y_pred[0, 1, :5] = 1.0
        result = compute_absolute_volume_difference(y_pred=y_pred, y=y_true, ignore_empty=False)
        self.assertAlmostEqual(result[0, 1].item(), 5.0)

    def test_shape_mismatch_raises(self):
        """Mismatched y_pred and y shapes should raise a ValueError."""
        with self.assertRaises(ValueError):
            compute_absolute_volume_difference(y_pred=torch.zeros(2, 3, 8, 8), y=torch.zeros(2, 3, 4, 4))

    def test_too_few_dims_raises(self):
        """Input tensors with fewer than 3 dimensions should raise a ValueError."""
        with self.assertRaises(ValueError):
            compute_absolute_volume_difference(y_pred=torch.zeros(2, 3), y=torch.zeros(2, 3))

    def test_3d_volumes(self):
        """AVD should correctly count voxel differences in 3-D spatial inputs."""
        # 3-D spatial (D, H, W)
        y_pred = torch.zeros(1, 2, 8, 8, 8)
        y_true = torch.zeros(1, 2, 8, 8, 8)
        y_pred[0, 1, :4, :4, :4] = 1.0  # 64 voxels
        y_true[0, 1, :3, :3, :3] = 1.0  # 27 voxels
        result = compute_absolute_volume_difference(y_pred=y_pred, y=y_true, ignore_empty=False)
        self.assertAlmostEqual(result[0, 1].item(), 37.0)

    def test_output_shape_multi_class(self):
        """Output shape should be [batch_size, num_classes] for multi-class inputs."""
        y = torch.randint(0, 2, (4, 5, 16, 16)).float()
        result = compute_absolute_volume_difference(y_pred=y, y=y, ignore_empty=False)
        self.assertEqual(result.shape, torch.Size([4, 5]))


class TestAbsoluteVolumeDifferenceMetric(unittest.TestCase):
    """Tests for the AbsoluteVolumeDifferenceMetric class (cumulative interface)."""

    def test_aggregate_mean(self):
        """Mean reduction over accumulated batches should return the correct per-class AVD."""
        y_pred = torch.zeros(2, 2, 8, 8)
        y_true = torch.zeros(2, 2, 8, 8)
        y_pred[:, 1, :6, :6] = 1.0  # 36 voxels per batch item
        y_true[:, 1, :4, :4] = 1.0  # 16 voxels per batch item
        metric = AbsoluteVolumeDifferenceMetric(include_background=False, reduction="mean", ignore_empty=False)
        metric(y_pred, y_true)
        agg = metric.aggregate()
        # single foreground channel, AVD = 20 for both batch items → mean = 20
        self.assertAlmostEqual(agg.item(), 20.0)
        metric.reset()

    def test_aggregate_returns_not_nans_when_requested(self):
        """When get_not_nans=True, aggregate should return a (metric, not_nans) tuple."""
        y_pred = torch.zeros(2, 2, 4, 4)
        y_true = torch.zeros(2, 2, 4, 4)
        y_pred[:, 1, :2, :2] = 1.0
        y_true[:, 1, :2, :2] = 1.0
        metric = AbsoluteVolumeDifferenceMetric(include_background=False, get_not_nans=True)
        metric(y_pred, y_true)
        result, not_nans = metric.aggregate()
        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(not_nans, torch.Tensor)
        metric.reset()

    def test_cumulative_accumulation(self):
        """Multiple forward calls before aggregate should use all accumulated data correctly."""
        # calling the metric twice and aggregating should use all accumulated data
        metric = AbsoluteVolumeDifferenceMetric(include_background=False, reduction="mean", ignore_empty=False)
        for _ in range(3):
            y_pred = torch.zeros(1, 2, 8)
            y_true = torch.zeros(1, 2, 8)
            y_pred[0, 1, :6] = 1.0
            y_true[0, 1, :4] = 1.0
            metric(y_pred, y_true)
        agg = metric.aggregate()
        self.assertAlmostEqual(agg.item(), 2.0)
        metric.reset()

    def test_reset_clears_buffer(self):
        """Calling reset() should clear the buffer so a subsequent aggregate() raises."""
        metric = AbsoluteVolumeDifferenceMetric(ignore_empty=False)
        y = torch.zeros(1, 2, 4)
        y[0, 1, :2] = 1.0
        metric(y, y)
        metric.reset()
        # after reset the buffer should be empty; calling aggregate raises
        with self.assertRaises(ValueError):
            metric.aggregate()


if __name__ == "__main__":
    unittest.main()
