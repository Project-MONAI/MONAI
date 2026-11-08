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
from unittest import mock

import torch
from parameterized import parameterized

from monai.metrics import CalibrationErrorMetric, CalibrationReduction, calibration_binning
from monai.utils import MetricReduction
from tests.test_utils import assert_allclose

_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Test cases for calibration binning
# Format: [name, y_pred, y, num_bins, right, expected_mean_p, expected_mean_gt, expected_counts]
TEST_BINNING_SMALL_MID = [
    "small_mid",
    torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]),
    torch.tensor([[[[1, 0], [0, 1]]]]),
    5,
    False,
    torch.tensor([[[0.1, 0.3, float("nan"), 0.7, 0.9]]]),
    torch.tensor([[[0.0, 0.0, float("nan"), 1.0, 1.0]]]),
    torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]]),
]

TEST_BINNING_LARGE_MID = [
    "large_mid",
    torch.tensor(
        [[[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]], [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]]]
    ),
    torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]]),
    5,
    False,
    torch.tensor(
        [
            [[0.1, 0.3, float("nan"), 0.7, 0.9], [float("nan"), 0.3, 0.5, 0.7, float("nan")]],
            [[float("nan"), 0.3, float("nan"), float("nan"), 0.9], [0.1, float("nan"), float("nan"), 0.7, 0.9]],
        ]
    ),
    torch.tensor(
        [
            [[0.0, 0.0, float("nan"), 1.0, 1.0], [float("nan"), 1.0, 0.5, 0.0, float("nan")]],
            [[float("nan"), 0.0, float("nan"), float("nan"), 1.0], [0.0, float("nan"), float("nan"), 1.0, 1.0]],
        ]
    ),
    torch.tensor(
        [[[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0, 0.0]], [[0.0, 2.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 1.0, 1.0]]]
    ),
]

TEST_BINNING_SMALL_LEFT_EDGE = [
    "small_left_edge",
    torch.tensor([[[[0.8, 0.2], [0.4, 0.6]]]]),
    torch.tensor([[[[1, 0], [0, 1]]]]),
    5,
    False,
    torch.tensor([[[0.2, 0.4, 0.6, 0.8, float("nan")]]]),
    torch.tensor([[[0.0, 0.0, 1.0, 1.0, float("nan")]]]),
    torch.tensor([[[1.0, 1.0, 1.0, 1.0, 0.0]]]),
]

TEST_BINNING_SMALL_RIGHT_EDGE = [
    "small_right_edge",
    torch.tensor([[[[0.8, 0.2], [0.4, 0.6]]]]),
    torch.tensor([[[[1, 0], [0, 1]]]]),
    5,
    True,
    torch.tensor([[[float("nan"), 0.2, 0.4, 0.6, 0.8]]]),
    torch.tensor([[[float("nan"), 0.0, 0.0, 1.0, 1.0]]]),
    torch.tensor([[[0.0, 1.0, 1.0, 1.0, 1.0]]]),
]

BINNING_TEST_CASES = [
    TEST_BINNING_SMALL_MID,
    TEST_BINNING_LARGE_MID,
    TEST_BINNING_SMALL_LEFT_EDGE,
    TEST_BINNING_SMALL_RIGHT_EDGE,
]

# Test cases for calibration error metric values
# Format: [name, y_pred, y, num_bins, expected_expected, expected_average, expected_maximum]
TEST_VALUE_1B1C = [
    "1b1c",
    torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]),
    torch.tensor([[[[1, 0], [0, 1]]]]),
    5,
    torch.tensor([[0.2]]),
    torch.tensor([[0.2]]),
    torch.tensor([[0.3]]),
]

TEST_VALUE_2B2C = [
    "2b2c",
    torch.tensor(
        [[[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]], [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]]]
    ),
    torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]]),
    5,
    torch.tensor([[0.2000, 0.3500], [0.2000, 0.1500]]),
    torch.tensor([[0.2000, 0.4667], [0.2000, 0.1667]]),
    torch.tensor([[0.3000, 0.7000], [0.3000, 0.3000]]),
]

VALUE_TEST_CASES = [TEST_VALUE_1B1C, TEST_VALUE_2B2C]


class TestCalibrationBinning(unittest.TestCase):

    @parameterized.expand(BINNING_TEST_CASES)
    def test_binning(self, _name, y_pred, y, num_bins, right, expected_mean_p, expected_mean_gt, expected_counts):
        y_pred = y_pred.to(_device)
        y = y.to(_device)
        expected_mean_p = expected_mean_p.to(_device)
        expected_mean_gt = expected_mean_gt.to(_device)
        expected_counts = expected_counts.to(_device)

        # Use mock.patch to replace torch.linspace
        # This is to avoid floating point precision issues when looking at edge conditions
        mock_boundaries = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=_device)
        with mock.patch("monai.metrics.calibration.torch.linspace", return_value=mock_boundaries):
            mean_p_per_bin, mean_gt_per_bin, bin_counts = calibration_binning(y_pred, y, num_bins=num_bins, right=right)

        # Handle NaN comparisons: compare NaN masks separately, then compare non-NaN values
        # mean_p_per_bin
        self.assertTrue(torch.equal(torch.isnan(mean_p_per_bin), torch.isnan(expected_mean_p)))
        mask_p = ~torch.isnan(expected_mean_p)
        if mask_p.any():
            assert_allclose(mean_p_per_bin[mask_p], expected_mean_p[mask_p], atol=1e-4, rtol=1e-4)

        # mean_gt_per_bin
        self.assertTrue(torch.equal(torch.isnan(mean_gt_per_bin), torch.isnan(expected_mean_gt)))
        mask_gt = ~torch.isnan(expected_mean_gt)
        if mask_gt.any():
            assert_allclose(mean_gt_per_bin[mask_gt], expected_mean_gt[mask_gt], atol=1e-4, rtol=1e-4)

        # bin_counts (no NaNs)
        assert_allclose(bin_counts, expected_counts, atol=1e-4, rtol=1e-4)

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise ValueError."""
        y_pred = torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1], [0, 0]]]]).to(_device)  # Different shape
        with self.assertRaises(ValueError) as context:
            calibration_binning(y_pred, y, num_bins=5)
        self.assertIn("same shape", str(context.exception))

    def test_insufficient_ndim_raises(self):
        """Test that tensors with ndim < 3 raise ValueError."""
        y_pred = torch.tensor([[0.7, 0.3]]).to(_device)  # Only 2D
        y = torch.tensor([[1, 0]]).to(_device)
        with self.assertRaises(ValueError) as context:
            calibration_binning(y_pred, y, num_bins=5)
        self.assertIn("ndim", str(context.exception))

    def test_invalid_num_bins_raises(self):
        """Test that num_bins < 1 raises ValueError."""
        y_pred = torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]]]]).to(_device)
        with self.assertRaises(ValueError) as context:
            calibration_binning(y_pred, y, num_bins=0)
        self.assertIn("num_bins", str(context.exception))


class TestCalibrationErrorMetricValue(unittest.TestCase):

    @parameterized.expand(VALUE_TEST_CASES)
    def test_expected_reduction(self, _name, y_pred, y, num_bins, expected_expected, _expected_average, _expected_max):
        y_pred = y_pred.to(_device)
        y = y.to(_device)
        expected_expected = expected_expected.to(_device)

        metric = CalibrationErrorMetric(
            num_bins=num_bins,
            include_background=True,
            calibration_reduction=CalibrationReduction.EXPECTED,
            metric_reduction=MetricReduction.NONE,
        )

        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()

        assert_allclose(result, expected_expected, atol=1e-4, rtol=1e-4)

    @parameterized.expand(VALUE_TEST_CASES)
    def test_average_reduction(self, _name, y_pred, y, num_bins, _expected_expected, expected_average, _expected_max):
        y_pred = y_pred.to(_device)
        y = y.to(_device)
        expected_average = expected_average.to(_device)

        metric = CalibrationErrorMetric(
            num_bins=num_bins,
            include_background=True,
            calibration_reduction=CalibrationReduction.AVERAGE,
            metric_reduction=MetricReduction.NONE,
        )

        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()

        assert_allclose(result, expected_average, atol=1e-4, rtol=1e-4)

    @parameterized.expand(VALUE_TEST_CASES)
    def test_maximum_reduction(self, _name, y_pred, y, num_bins, _expected_expected, _expected_average, expected_max):
        y_pred = y_pred.to(_device)
        y = y.to(_device)
        expected_max = expected_max.to(_device)

        metric = CalibrationErrorMetric(
            num_bins=num_bins,
            include_background=True,
            calibration_reduction=CalibrationReduction.MAXIMUM,
            metric_reduction=MetricReduction.NONE,
        )

        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()

        assert_allclose(result, expected_max, atol=1e-4, rtol=1e-4)


class TestCalibrationErrorEmptyBins(unittest.TestCase):
    """Test edge cases when all bins are empty (division by zero scenarios)."""

    def test_expected_reduction_all_empty_bins_returns_nan(self):
        """Test that EXPECTED reduction returns NaN when all bins are empty (division by zero case)."""
        from unittest import mock

        y_pred = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]).to(_device)
        y = torch.tensor([[[[1, 1], [1, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=True,
            calibration_reduction=CalibrationReduction.EXPECTED,
            metric_reduction=MetricReduction.NONE,
        )

        # Mock calibration_binning to return zero bin_counts (all empty bins)
        def mock_binning(y_pred, y, num_bins, right):
            batch_size, num_channels = y_pred.shape[:2]
            device = y_pred.device
            mean_p = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            mean_gt = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            counts = torch.zeros((batch_size, num_channels, num_bins), device=device)
            return mean_p, mean_gt, counts

        with mock.patch("monai.metrics.calibration.calibration_binning", side_effect=mock_binning):
            metric(y_pred=y_pred, y=y)
            result = metric.aggregate()

        # All bins empty should result in NaN
        self.assertTrue(torch.isnan(result).all(), "Result should be NaN when all bins are empty")

    def test_average_reduction_all_empty_bins_returns_nan(self):
        """Test that AVERAGE reduction returns NaN when all bins are empty."""
        from unittest import mock

        y_pred = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]).to(_device)
        y = torch.tensor([[[[1, 1], [1, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=True,
            calibration_reduction=CalibrationReduction.AVERAGE,
            metric_reduction=MetricReduction.NONE,
        )

        def mock_binning(y_pred, y, num_bins, right):
            batch_size, num_channels = y_pred.shape[:2]
            device = y_pred.device
            mean_p = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            mean_gt = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            counts = torch.zeros((batch_size, num_channels, num_bins), device=device)
            return mean_p, mean_gt, counts

        with mock.patch("monai.metrics.calibration.calibration_binning", side_effect=mock_binning):
            metric(y_pred=y_pred, y=y)
            result = metric.aggregate()

        self.assertTrue(torch.isnan(result).all(), "Result should be NaN when all bins are empty")

    def test_maximum_reduction_all_empty_bins_returns_nan(self):
        """Test that MAXIMUM reduction returns NaN when all bins are empty."""
        from unittest import mock

        y_pred = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]).to(_device)
        y = torch.tensor([[[[1, 1], [1, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=True,
            calibration_reduction=CalibrationReduction.MAXIMUM,
            metric_reduction=MetricReduction.NONE,
        )

        def mock_binning(y_pred, y, num_bins, right):
            batch_size, num_channels = y_pred.shape[:2]
            device = y_pred.device
            mean_p = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            mean_gt = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            counts = torch.zeros((batch_size, num_channels, num_bins), device=device)
            return mean_p, mean_gt, counts

        with mock.patch("monai.metrics.calibration.calibration_binning", side_effect=mock_binning):
            metric(y_pred=y_pred, y=y)
            result = metric.aggregate()

        self.assertTrue(torch.isnan(result).all(), "Result should be NaN when all bins are empty")

    def test_expected_reduction_with_zeros_only_returns_nan(self):
        """Test EXPECTED reduction returns NaN for channels where all bin_counts are zero.

        This tests the actual division-by-zero fix: if we have values that all fall
        outside the valid probability range [0, 1], all bins would be empty.
        """
        # Create a 2-channel tensor where one channel has valid data and one is out of range
        # Note: calibration_binning clamps values, but we can test with very extreme distributions
        # that result in some channels having all NaN abs_diff
        # A simpler test: create data where bin_counts sum to zero for a channel

        # Use mock to simulate the scenario where bin_counts are zero for one channel
        from unittest import mock

        y_pred = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]]).to(_device)
        y = torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=True,
            calibration_reduction=CalibrationReduction.EXPECTED,
            metric_reduction=MetricReduction.NONE,
        )

        # Mock calibration_binning to return zero bin_counts for first channel
        def mock_binning(y_pred, y, num_bins, right):
            batch_size, num_channels = y_pred.shape[:2]
            device = y_pred.device

            # Create normal results for channel 1, all zeros for channel 0
            mean_p = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            mean_gt = torch.full((batch_size, num_channels, num_bins), float("nan"), device=device)
            counts = torch.zeros((batch_size, num_channels, num_bins), device=device)

            # Channel 1 has some data
            mean_p[0, 1, 2] = 0.5
            mean_gt[0, 1, 2] = 0.6
            counts[0, 1, 2] = 4.0

            return mean_p, mean_gt, counts

        with mock.patch("monai.metrics.calibration.calibration_binning", side_effect=mock_binning):
            metric(y_pred=y_pred, y=y)
            result = metric.aggregate()

        # Channel 0 should be NaN (all bins empty), Channel 1 should have a value
        self.assertTrue(torch.isnan(result[0, 0]).item(), "Channel 0 should be NaN when all bins are empty")
        self.assertFalse(torch.isnan(result[0, 1]).item(), "Channel 1 should have a valid value")
        assert_allclose(result[0, 1], torch.tensor(0.1, device=_device), atol=1e-4, rtol=1e-4)


class TestCalibrationErrorMetricOptions(unittest.TestCase):

    def test_include_background_false(self):
        y_pred = torch.tensor(
            [[[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]], [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]]]
        ).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=False,
            calibration_reduction=CalibrationReduction.EXPECTED,
            metric_reduction=MetricReduction.MEAN,
        )

        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()

        assert_allclose(result, torch.tensor(0.2500, device=_device), atol=1e-4, rtol=1e-4)

    def test_metric_reduction_mean(self):
        y_pred = torch.tensor(
            [[[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]], [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]]]
        ).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=True,
            calibration_reduction=CalibrationReduction.EXPECTED,
            metric_reduction=MetricReduction.MEAN,
        )

        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()

        # Mean of [[0.2000, 0.3500], [0.2000, 0.1500]] = 0.225
        assert_allclose(result, torch.tensor(0.2250, device=_device), atol=1e-4, rtol=1e-4)

    def test_get_not_nans(self):
        y_pred = torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=True,
            calibration_reduction=CalibrationReduction.EXPECTED,
            metric_reduction=MetricReduction.MEAN,
            get_not_nans=True,
        )

        metric(y_pred=y_pred, y=y)
        result, not_nans = metric.aggregate()

        assert_allclose(result, torch.tensor(0.2, device=_device), atol=1e-4, rtol=1e-4)
        self.assertEqual(not_nans.item(), 1)

    def test_cumulative_iterations(self):
        """Test that the metric correctly accumulates over multiple iterations."""
        y_pred = torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]]]]).to(_device)

        metric = CalibrationErrorMetric(
            num_bins=5,
            include_background=True,
            calibration_reduction=CalibrationReduction.EXPECTED,
            metric_reduction=MetricReduction.MEAN,
        )

        # First iteration
        metric(y_pred=y_pred, y=y)
        # Second iteration
        metric(y_pred=y_pred, y=y)

        result = metric.aggregate()
        # Should still be 0.2 since both iterations have the same data
        assert_allclose(result, torch.tensor(0.2, device=_device), atol=1e-4, rtol=1e-4)

        # Test reset
        metric.reset()
        data = metric.get_buffer()
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
