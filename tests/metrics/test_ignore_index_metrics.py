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
    compute_dice,
    compute_iou,
    get_confusion_matrix,
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


class TestIgnoreIndexMetrics(unittest.TestCase):
    @parameterized.expand(TEST_METRICS)
    def test_metric_ignore_consistency(self, metric_class, kwargs):
        metric = metric_class(ignore_index=255, **kwargs)

        y_pred1 = torch.zeros((1, 2, 4, 4))
        y_pred1[:, 1, 0:2, :] = 1.0

        y_pred2 = y_pred1.clone()
        y_pred2[:, 1, 2:4, :] = 1.0

        y = torch.zeros((1, 2, 4, 4))
        y[:, 1, 0:2, 0:2] = 1.0
        y[:, 0, 0:2, 2:4] = 1.0

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

    @parameterized.expand(
        [(metric_class, kwargs, ignore_index) for metric_class, kwargs in TEST_METRICS for ignore_index in (0, 1)]
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

    def test_ignored_voxels_excluded_from_other_classes(self):
        """Ignored voxels must be dropped from every class score, not just their own."""
        # 4 voxels, 3 one-hot classes; voxel 1 belongs to the ignored class 1
        y = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        # a perfect prediction except the ignored voxel is called class 0
        y_pred = torch.tensor([[[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])

        iou = compute_iou(y_pred, y, include_background=True, ignore_index=1)
        dice = compute_dice(y_pred, y, include_background=True, ignore_index=1)

        # the mislabelled voxel is ignored, so class 0 is scored as perfect
        self.assertEqual(iou[0, 0].item(), 1.0)
        torch.testing.assert_close(iou, dice, equal_nan=True)

    @parameterized.expand([(background, sample) for background in (True, False) for sample in (True, False)])
    def test_confusion_matrix_excludes_ignored_voxels(self, include_background, compute_sample):
        """Exclude ignored-class voxels from counts and precision in both aggregation modes.

        Args:
            include_background: whether to include the background channel in the output.
            compute_sample: whether to compute precision per sample before aggregation.
        """
        y = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        # The only error predicts class 2 at a voxel belonging to ignored class 1.
        y_pred = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
        metric = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name="precision",
            compute_sample=compute_sample,
            reduction="none",
            ignore_index=1,
        )
        metric(y_pred=y_pred, y=y)
        class_index = 2 if include_background else 1
        self.assertEqual(metric.aggregate()[0][0, class_index].item(), 1.0)
        counts = get_confusion_matrix(y_pred, y, include_background=include_background, ignore_index=1)
        self.assertEqual(counts[0, class_index].tolist(), [1.0, 0.0, 2.0, 0.0])

    @parameterized.expand([(True,), (False,)])
    def test_confusion_matrix_excludes_ignored_background_from_negatives(self, include_background):
        """Exclude ignored background voxels from every channel's negative counts.

        Args:
            include_background: whether to include the background channel in the output.
        """
        y = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        y_pred = y.clone()
        y_pred[0, 0, 0] = 0.0
        y_pred[0, 1, 0] = 1.0
        counts = get_confusion_matrix(y_pred, y, include_background=include_background, ignore_index=0)
        class_index = 2 if include_background else 1
        self.assertEqual(counts[0, class_index].tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_confusion_matrix_keeps_errors_on_valid_voxels(self):
        """Preserve false positives and false negatives at valid voxels."""
        y = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        y_pred = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
        counts = get_confusion_matrix(y_pred, y, include_background=False, ignore_index=0)
        self.assertEqual(counts[0, 0].tolist(), [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(counts[0, 1].tolist(), [1.0, 1.0, 0.0, 0.0])

    @parameterized.expand([(True,), (False,)])
    def test_confusion_matrix_sentinel_mask(self, include_background):
        """Preserve sentinel masking with and without the background channel.

        Args:
            include_background: whether to include the background channel in the output.
        """
        y = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
        y_pred = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
        counts = get_confusion_matrix(y_pred, y, include_background=include_background, ignore_index=255)
        class_index = 2 if include_background else 1
        self.assertEqual(counts[0, class_index].tolist(), [1.0, 0.0, 2.0, 0.0])

    @parameterized.expand(
        [
            (background, ignore_index, dtype)
            for background in (True, False)
            for ignore_index in (0, 1, 2)
            for dtype in (torch.float32, torch.int64, torch.bool)
        ]
    )
    def test_confusion_matrix_multichannel_boolean(self, include_background, ignore_index, prediction_dtype):
        """Exclude ignored voxels from boolean one-hot targets without mutating inputs.

        Args:
            include_background: whether to retain the background channel in the returned counts.
            ignore_index: original class channel whose spatial locations must be excluded.
            prediction_dtype: data type of the binarized prediction tensor.
        """
        y = torch.tensor([[[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=torch.bool)
        y_pred = torch.tensor([[[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]]], dtype=prediction_dtype)
        original_y, original_pred = y.clone(), y_pred.clone()
        expected = torch.tensor(
            [
                [[0, 0, 2, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
                [[2, 0, 1, 0], [0, 0, 3, 0], [1, 0, 2, 0]],
                [[2, 1, 0, 0], [0, 0, 2, 1], [0, 0, 3, 0]],
            ],
            dtype=torch.float32,
        )[ignore_index].unsqueeze(0)
        if not include_background:
            expected = expected[:, 1:]

        counts = get_confusion_matrix(y_pred, y, include_background=include_background, ignore_index=ignore_index)

        torch.testing.assert_close(counts, expected)
        torch.testing.assert_close(y, original_y)
        torch.testing.assert_close(y_pred, original_pred)

    @parameterized.expand(
        [(background, dtype) for background in (True, False) for dtype in (torch.float32, torch.int64, torch.bool)]
    )
    def test_confusion_matrix_single_channel_ignore_class(self, include_background, dtype):
        """Interpret a single channel as binary membership when ignoring class zero.

        Args:
            include_background: whether to request inclusion of the background channel.
            dtype: data type of the binary target and prediction tensors.
        """
        y = torch.tensor([[[1, 1, 0]]], dtype=dtype)
        y_pred = torch.ones_like(y)
        counts = get_confusion_matrix(y_pred, y, include_background=include_background, ignore_index=0)
        self.assertEqual(counts[0, 0].tolist(), [0.0, 1.0, 0.0, 0.0])

    @parameterized.expand([(True,), (False,)])
    def test_confusion_matrix_single_channel_specificity(self, include_background):
        """Retain a false positive at the only valid single-channel voxel.

        Args:
            include_background: whether to request inclusion of the background channel.
        """
        y = torch.tensor([[[1.0, 1.0, 0.0]]])
        metric = ConfusionMatrixMetric(
            include_background=include_background, metric_name="specificity", reduction="none", ignore_index=0
        )
        metric(y_pred=torch.ones_like(y), y=y)
        self.assertEqual(metric.aggregate()[0].item(), 0.0)

    @parameterized.expand(
        [
            (True, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
            (False, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
            (True, [0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]),
            (False, [0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]),
        ]
    )
    def test_confusion_matrix_single_channel_mask_extremes(self, include_background, labels, expected):
        """Handle all-ignored and entirely valid single-channel spatial regions.

        Args:
            include_background: whether to request inclusion of the background channel.
            labels: binary target values for the spatial locations.
            expected: expected counts in true positive, false positive, true negative, false negative order.
        """
        y = torch.tensor([[labels]])
        counts = get_confusion_matrix(torch.ones_like(y), y, include_background=include_background, ignore_index=0)
        self.assertEqual(counts[0, 0].tolist(), expected)

    @parameterized.expand([(True,), (False,)])
    def test_confusion_matrix_single_channel_sentinel(self, include_background):
        """Keep the existing out-of-range comparison for single-channel targets.

        Args:
            include_background: whether to request inclusion of the background channel.
        """
        y = torch.tensor([[[1.0, 1.0, 0.0]]])
        counts = get_confusion_matrix(torch.ones_like(y), y, include_background=include_background, ignore_index=255)
        self.assertEqual(counts[0, 0].tolist(), [2.0, 1.0, 0.0, 0.0])

    @parameterized.expand([(True,), (False,)])
    def test_confusion_matrix_single_channel_without_ignore(self, include_background):
        """Leave the unmasked single-channel counts unchanged.

        Args:
            include_background: whether to request inclusion of the background channel.
        """
        y = torch.tensor([[[1.0, 1.0, 0.0]]])
        counts = get_confusion_matrix(torch.ones_like(y), y, include_background=include_background)
        self.assertEqual(counts[0, 0].tolist(), [2.0, 1.0, 0.0, 0.0])

    def test_ignored_voxels_excluded_with_include_background_false(self):
        """The ignore_index mask must line up with the ignore_background channel strip."""
        # 4 one-hot classes: 0=background, 1, 2=ignored, 3
        y = torch.zeros(1, 4, 4)
        y[0, 0, 0] = 1  # voxel 0 -> background
        y[0, 2, 1] = 1  # voxel 1 -> ignored class
        y[0, 1, 2] = 1  # voxel 2 -> class 1
        y[0, 3, 3] = 1  # voxel 3 -> class 3

        y_pred = y.clone()
        # mislabel the ignored voxel as class 1 instead of leaving it unpredicted
        y_pred[0, 2, 1] = 0
        y_pred[0, 1, 1] = 1

        iou = compute_iou(y_pred, y, include_background=False, ignore_index=2)
        dice = compute_dice(y_pred, y, include_background=False, ignore_index=2)

        # class 1's false positive at the ignored voxel must be dropped, not just
        # its own (now background-stripped) channel
        self.assertEqual(iou[0, 0].item(), 1.0)
        torch.testing.assert_close(iou, dice, equal_nan=True)


@unittest.skipUnless(has_scipy, "Scipy required for surface metrics")
class TestIgnoreIndexSurfaceMetrics(unittest.TestCase):
    @parameterized.expand(SCIPY_METRICS)
    def test_metric_ignore_consistency(self, metric_class, kwargs):
        metric = metric_class(ignore_index=255, **kwargs)

        y_pred1 = torch.zeros((1, 2, 4, 4))
        y_pred1[:, 1, 0:2, :] = 1.0

        y_pred2 = y_pred1.clone()
        y_pred2[:, 1, 2:4, :] = 1.0

        y = torch.zeros((1, 2, 4, 4))
        y[:, 1, 0:2, 0:2] = 1.0
        y[:, 0, 0:2, 2:4] = 1.0

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

    @parameterized.expand(
        [(metric_class, kwargs, ignore_index) for metric_class, kwargs in SCIPY_METRICS for ignore_index in (0, 1)]
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


if __name__ == "__main__":
    unittest.main()
