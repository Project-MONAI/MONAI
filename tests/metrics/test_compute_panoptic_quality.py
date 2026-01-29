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

from monai.metrics import PanopticQualityMetric, compute_panoptic_quality
from monai.metrics.panoptic_quality import compute_mean_iou
from tests.test_utils import SkipIfNoModule

_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# TEST_FUNC_CASE related cases are used to test for single image with HW input shape

sample_1 = torch.randint(low=0, high=5, size=(64, 64), device=_device)
sample_2_pred = torch.as_tensor([[0, 1, 1, 1], [0, 0, 0, 0], [2, 0, 3, 3], [4, 2, 2, 0]], device=_device)
sample_2_pred_need_remap = torch.as_tensor([[0, 7, 7, 7], [0, 0, 0, 0], [1, 0, 8, 8], [9, 1, 1, 0]], device=_device)
sample_2_gt = torch.as_tensor([[1, 1, 2, 1], [0, 0, 0, 0], [1, 3, 0, 0], [4, 3, 3, 3]], device=_device)
# if pred == gt, result should be 1
TEST_FUNC_CASE_1 = [{"pred": sample_1, "gt": sample_1, "match_iou_threshold": 0.99}, 1.0]

# test sample_2 when match_iou_threshold = 0.5
TEST_FUNC_CASE_2 = [{"pred": sample_2_pred, "gt": sample_2_gt, "match_iou_threshold": 0.5}, 0.25]
# test sample_2 when match_iou_threshold = 0.3, metric_name = "sq"
TEST_FUNC_CASE_3 = [{"pred": sample_2_pred, "gt": sample_2_gt, "metric_name": "sq", "match_iou_threshold": 0.3}, 0.6]
# test sample_2 when match_iou_threshold = 0.3, pred has different order, metric_name = "RQ"
TEST_FUNC_CASE_4 = [
    {"pred": sample_2_pred_need_remap, "gt": sample_2_gt, "metric_name": "RQ", "match_iou_threshold": 0.3},
    0.75,
]

# TEST_CLS_CASE related cases are used to test the PanopticQualityMetric with B2HW input
sample_3_pred = torch.as_tensor(
    [
        [[[2, 0, 1], [2, 1, 1], [0, 1, 1]], [[0, 1, 3], [0, 0, 0], [1, 2, 1]]],
        [[[1, 1, 1], [3, 2, 0], [3, 2, 1]], [[1, 1, 3], [3, 1, 1], [0, 3, 0]]],
    ],
    device=_device,
)

sample_3_gt = torch.as_tensor(
    [
        [[[2, 0, 0], [2, 0, 0], [2, 2, 3]], [[3, 3, 3], [3, 2, 1], [2, 2, 3]]],
        [[[1, 1, 1], [0, 0, 3], [0, 0, 3]], [[0, 1, 3], [2, 1, 0], [3, 0, 3]]],
    ],
    device=_device,
)

# test sample_3, num_classes = 3, match_iou_threshold = 0.5
TEST_CLS_CASE_1 = [{"num_classes": 3, "match_iou_threshold": 0.5}, sample_3_pred, sample_3_gt, (0.0, 0.0, 0.25)]

# test sample_3, num_classes = 3, match_iou_threshold = 0.3
TEST_CLS_CASE_2 = [{"num_classes": 3, "match_iou_threshold": 0.3}, sample_3_pred, sample_3_gt, (0.25, 0.5, 0.25)]

# test sample_3, num_classes = 4, match_iou_threshold = 0.3, metric_name = "segmentation_quality"
TEST_CLS_CASE_3 = [
    {"num_classes": 4, "match_iou_threshold": 0.3, "metric_name": "segmentation_quality"},
    sample_3_pred,
    sample_3_gt,
    (0.5, 0.5, 1.0, 0.0),
]

# test sample_3, num_classes = 3, match_iou_threshold = 0.4, reduction = "none", metric_name = "Recognition Quality"
TEST_CLS_CASE_4 = [
    {"num_classes": 3, "reduction": "none", "match_iou_threshold": 0.4, "metric_name": "Recognition Quality"},
    sample_3_pred,
    sample_3_gt,
    [[0.0, 1.0, 0.0], [0.6667, 0.0, 0.4]],
]

# test sample_3, num_classes = 3, match_iou_threshold = 0.4, reduction = "none", multiple metrics
TEST_CLS_CASE_5 = [
    {"num_classes": 3, "reduction": "none", "match_iou_threshold": 0.4, "metric_name": ["Recognition Quality", "pq"]},
    sample_3_pred,
    sample_3_gt,
    [torch.as_tensor([[0.0, 1.0, 0.0], [0.6667, 0.0, 0.4]]), torch.as_tensor([[0.0, 0.5, 0.0], [0.3333, 0.0, 0.4]])],
]

# 3D test cases
sample_3d_pred = torch.as_tensor(
    [[[[[2, 0], [1, 1]], [[0, 1], [2, 1]]], [[[0, 1], [3, 0]], [[1, 0], [1, 1]]]]],  # instance channel  # class channel
    device=_device,
)

sample_3d_gt = torch.as_tensor(
    [[[[[2, 0], [0, 0]], [[2, 2], [2, 3]]], [[[3, 3], [3, 2]], [[2, 2], [3, 3]]]]],  # instance channel  # class channel
    device=_device,
)

# test 3D sample, num_classes = 3, match_iou_threshold = 0.5
TEST_3D_CASE_1 = [{"num_classes": 3, "match_iou_threshold": 0.5}, sample_3d_pred, sample_3d_gt]

# test confusion matrix return
TEST_CM_CASE_1 = [
    {"num_classes": 3, "match_iou_threshold": 0.5, "return_confusion_matrix": True},
    sample_3_pred,
    sample_3_gt,
]


@SkipIfNoModule("scipy.optimize")
class TestPanopticQualityMetric(unittest.TestCase):
    @parameterized.expand([TEST_FUNC_CASE_1, TEST_FUNC_CASE_2, TEST_FUNC_CASE_3, TEST_FUNC_CASE_4])
    def test_value(self, input_params, expected_value):
        result = compute_panoptic_quality(**input_params)
        np.testing.assert_allclose(result.cpu().detach().item(), expected_value, atol=1e-4)
        np.testing.assert_equal(result.device, input_params["pred"].device)

    @parameterized.expand([TEST_CLS_CASE_1, TEST_CLS_CASE_2, TEST_CLS_CASE_3, TEST_CLS_CASE_4, TEST_CLS_CASE_5])
    def test_value_class(self, input_params, y_pred, y_gt, expected_value):
        metric = PanopticQualityMetric(**input_params)
        metric(y_pred, y_gt)
        outputs = metric.aggregate()
        if isinstance(outputs, list):
            for output, value in zip(outputs, expected_value):
                np.testing.assert_allclose(output.cpu().numpy(), np.asarray(value), atol=1e-4)
        else:
            np.testing.assert_allclose(outputs.cpu().numpy(), np.asarray(expected_value), atol=1e-4)

    def test_3d_support(self):
        """Test that 3D input is properly supported."""
        input_params, y_pred, y_gt = TEST_3D_CASE_1
        metric = PanopticQualityMetric(**input_params)
        # Should not raise an error for 3D input
        metric(y_pred, y_gt)
        outputs = metric.aggregate()
        # Check that output is a tensor
        self.assertIsInstance(outputs, torch.Tensor)
        # Check that output shape is correct (num_classes,)
        self.assertEqual(outputs.shape, torch.Size([3]))

    def test_confusion_matrix_return(self):
        """Test that confusion matrix can be returned instead of computed metrics."""
        input_params, y_pred, y_gt = TEST_CM_CASE_1
        metric = PanopticQualityMetric(**input_params)
        metric(y_pred, y_gt)
        outputs = metric.aggregate()
        # Check that output is a tensor with shape (batch_size, num_classes, 4)
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape[-1], 4)
        # Verify that values correspond to [tp, fp, fn, iou_sum]
        tp, fp, fn, iou_sum = outputs[..., 0], outputs[..., 1], outputs[..., 2], outputs[..., 3]
        # tp, fp, fn should be non-negative integers
        self.assertTrue(torch.all(tp >= 0))
        self.assertTrue(torch.all(fp >= 0))
        self.assertTrue(torch.all(fn >= 0))
        # iou_sum should be non-negative float
        self.assertTrue(torch.all(iou_sum >= 0))

    def test_compute_mean_iou(self):
        """Test mean IoU computation from confusion matrix."""
        input_params, y_pred, y_gt = TEST_CM_CASE_1
        metric = PanopticQualityMetric(**input_params)
        metric(y_pred, y_gt)
        confusion_matrix = metric.aggregate()
        mean_iou = compute_mean_iou(confusion_matrix)

        # Check shape is correct
        self.assertEqual(mean_iou.shape, confusion_matrix.shape[:-1])

        # Check values are non-negative
        self.assertTrue(torch.all(mean_iou >= 0))

        # Validate against expected values
        # mean_iou = iou_sum / (tp + smooth_numerator)
        tp = confusion_matrix[..., 0]
        iou_sum = confusion_matrix[..., 3]
        expected_mean_iou = iou_sum / (tp + 1e-6)  # smooth_numerator=1e-6 is default
        np.testing.assert_allclose(mean_iou.cpu().numpy(), expected_mean_iou.cpu().numpy(), atol=1e-4)

    def test_metric_name_filtering(self):
        """Test that metric_name parameter properly filters output."""
        # Test single metric "sq"
        metric_sq = PanopticQualityMetric(num_classes=3, metric_name="sq", match_iou_threshold=0.5)
        metric_sq(sample_3_pred, sample_3_gt)
        result_sq = metric_sq.aggregate()
        self.assertIsInstance(result_sq, torch.Tensor)
        self.assertEqual(result_sq.shape, torch.Size([3]))

        # Test single metric "rq"
        metric_rq = PanopticQualityMetric(num_classes=3, metric_name="rq", match_iou_threshold=0.5)
        metric_rq(sample_3_pred, sample_3_gt)
        result_rq = metric_rq.aggregate()
        self.assertIsInstance(result_rq, torch.Tensor)
        self.assertEqual(result_rq.shape, torch.Size([3]))

        # Results should be different for different metrics
        self.assertFalse(torch.allclose(result_sq, result_rq, atol=1e-4))

    def test_invalid_3d_shape(self):
        """Test that invalid 3D shapes are rejected."""
        # Shape with 3 dimensions should fail
        invalid_pred = torch.randint(0, 5, (2, 2, 10))
        invalid_gt = torch.randint(0, 5, (2, 2, 10))
        metric = PanopticQualityMetric(num_classes=3)
        with self.assertRaises(ValueError):
            metric(invalid_pred, invalid_gt)

        # Shape with 6 dimensions should fail
        invalid_pred = torch.randint(0, 5, (1, 2, 8, 8, 8, 8))
        invalid_gt = torch.randint(0, 5, (1, 2, 8, 8, 8, 8))
        with self.assertRaises(ValueError):
            metric(invalid_pred, invalid_gt)

    def test_compute_mean_iou_invalid_shape(self):
        """Test that compute_mean_iou raises ValueError for invalid shapes."""
        from monai.metrics.panoptic_quality import compute_mean_iou

        # Shape (..., 3) instead of (..., 4) should fail
        invalid_confusion_matrix = torch.zeros(3, 3)
        with self.assertRaises(ValueError):
            compute_mean_iou(invalid_confusion_matrix)

        # Shape (..., 5) should also fail
        invalid_confusion_matrix = torch.zeros(2, 5)
        with self.assertRaises(ValueError):
            compute_mean_iou(invalid_confusion_matrix)


if __name__ == "__main__":
    unittest.main()
