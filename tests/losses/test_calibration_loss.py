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
import warnings

import torch

from monai.losses import HardL1ACELoss, SoftL1ACELoss
from monai.losses.calibration import _hard_binned_calibration, _soft_binned_calibration
from monai.metrics import calibration_binning
from tests.test_utils import skip_if_no_cuda
from tests.test_utils import test_script_save as check_script_save


class TestCalibrationBinningHelpers(unittest.TestCase):
    def test_hard_matches_metric_at_boundaries(self):
        prediction = torch.tensor([[[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]])
        target = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]])
        for right in (False, True):
            with self.subTest(right=right):
                sum_p, sum_target, counts = _hard_binned_calibration(prediction, target, 5, right)
                metric_p, metric_target, metric_counts = calibration_binning(prediction, target, 5, right)
                valid = counts > 0
                torch.testing.assert_close(counts, metric_counts)
                torch.testing.assert_close(sum_p[valid] / counts[valid], metric_p[valid])
                torch.testing.assert_close(sum_target[valid] / counts[valid], metric_target[valid])

    def test_soft_exact_interpolation(self):
        prediction = torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0]]], dtype=torch.float64)
        target = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 1.0]]], dtype=torch.float64)
        sum_p, sum_target, counts = _soft_binned_calibration(prediction, target, 2, False)
        torch.testing.assert_close(counts, torch.tensor([[[2.5, 2.5]]], dtype=torch.float64))
        torch.testing.assert_close(sum_p, torch.tensor([[[0.5, 2.0]]], dtype=torch.float64), rtol=0, atol=3e-7)
        torch.testing.assert_close(sum_target, torch.tensor([[[0.5, 2.5]]], dtype=torch.float64), rtol=0, atol=3e-7)

    def test_soft_boundary_rule_and_continuity(self):
        target = torch.tensor([[[1.0]]], dtype=torch.float64)
        at_center = torch.tensor([[[0.3750000447034836]]], dtype=torch.float64)
        left = at_center - 1e-7
        right = at_center + 1e-7
        for boundary_rule in (False, True):
            with self.subTest(right=boundary_rule):
                center_stats = _soft_binned_calibration(at_center, target, 4, boundary_rule)
                left_stats = _soft_binned_calibration(left, target, 4, boundary_rule)
                right_stats = _soft_binned_calibration(right, target, 4, boundary_rule)
                self.assertAlmostEqual(center_stats[2].sum().item(), 1.0)
                self.assertLess(torch.max(torch.abs(left_stats[2] - center_stats[2])).item(), 1e-6)
                self.assertLess(torch.max(torch.abs(right_stats[2] - center_stats[2])).item(), 1e-6)


class TestCalibrationLoss(unittest.TestCase):
    loss_types = (HardL1ACELoss, SoftL1ACELoss)

    def test_manual_loss_values(self):
        prediction = torch.tensor([[[0.1, 0.3, 0.7, 0.9]]])
        target = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        result = HardL1ACELoss(num_bins=5)(prediction, target)
        torch.testing.assert_close(result, torch.tensor(0.2))

        prediction = torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0]]], dtype=torch.float64)
        target = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 1.0]]], dtype=torch.float64)
        result = SoftL1ACELoss(num_bins=2)(prediction, target)
        torch.testing.assert_close(result, torch.tensor(0.1, dtype=torch.float64))

    def test_calibrated_and_miscalibrated_values(self):
        target = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                loss = loss_type(num_bins=4)
                torch.testing.assert_close(loss(target, target), torch.tensor(0.0))
                self.assertGreater(loss(1.0 - target, target).item(), 0.0)
                underconfident = target * 0.5 + 0.25
                self.assertGreater(loss(underconfident, target).item(), 0.0)

    def test_batch_class_reductions_and_weights(self):
        prediction = torch.tensor([[[0.1, 0.9], [0.2, 0.8]], [[0.3, 0.7], [0.4, 0.6]]])
        target = torch.tensor([[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]])
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                none = loss_type(num_bins=5, reduction="none")(prediction, target)
                self.assertEqual(none.shape, (2, 2, 1))
                torch.testing.assert_close(loss_type(num_bins=5)(prediction, target), none.mean())
                torch.testing.assert_close(loss_type(num_bins=5, reduction="sum")(prediction, target), none.sum())
                torch.testing.assert_close(loss_type(num_bins=5, weight=2.0)(prediction, target), 2.0 * none.mean())
                weighted = none * torch.tensor([1.0, 3.0]).view(1, 2, 1)
                torch.testing.assert_close(
                    loss_type(num_bins=5, weight=[1.0, 3.0])(prediction, target), weighted.mean()
                )

    def test_empty_class_is_zero_before_mean(self):
        prediction = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]])
        target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                none = loss_type(num_bins=5, reduction="none", ignore_empty_classes=True)(prediction, target)
                torch.testing.assert_close(none[:, 1], torch.zeros_like(none[:, 1]))
                torch.testing.assert_close(
                    loss_type(num_bins=5, ignore_empty_classes=True)(prediction, target), none.mean()
                )
                torch.testing.assert_close(none[:, 0].mean(), 2 * none.mean())
                included = loss_type(num_bins=5, ignore_empty_classes=False)(prediction, target)
                self.assertGreater(included.item(), 0.0)

    def test_every_class_empty_returns_differentiable_zero(self):
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                prediction = torch.full((2, 2, 4), 0.25, requires_grad=True)
                target = torch.zeros_like(prediction)
                result = loss_type(ignore_empty_classes=True)(prediction, target)
                torch.testing.assert_close(result, torch.tensor(0.0))
                result.backward()
                self.assertIsNotNone(prediction.grad)
                torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction.grad))

    def test_background_exclusion_and_shape(self):
        prediction = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                result = loss_type(include_background=False, reduction="none")(prediction, target)
                self.assertEqual(result.shape, (1, 1, 1))

    def test_activation_and_one_hot(self):
        logits = torch.tensor([[[0.2, -0.4], [-0.2, 0.4]]])
        labels = torch.tensor([[[0, 1]]])
        one_hot_target = torch.nn.functional.one_hot(labels[:, 0], num_classes=2).movedim(-1, 1).float()
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                expected_softmax = loss_type()(torch.softmax(logits, 1), one_hot_target)
                torch.testing.assert_close(loss_type(softmax=True, to_onehot_y=True)(logits, labels), expected_softmax)
                expected_sigmoid = loss_type()(torch.sigmoid(logits[:, :1]), one_hot_target[:, :1])
                torch.testing.assert_close(
                    loss_type(sigmoid=True)(logits[:, :1], one_hot_target[:, :1]), expected_sigmoid
                )
                torch.testing.assert_close(
                    loss_type(other_act=lambda value: value.square())(logits.abs(), one_hot_target),
                    loss_type()(logits.square(), one_hot_target),
                )

    def test_invalid_options_and_shapes(self):
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                with self.assertRaises(ValueError):
                    loss_type(num_bins=0)
                with self.assertRaises(ValueError):
                    loss_type(sigmoid=True, softmax=True)
                with self.assertRaises(TypeError):
                    loss_type(other_act=1)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    loss_type(weight=[1.0, -1.0])
                for invalid_weight in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(invalid_weight=invalid_weight):
                        with self.assertRaises(ValueError):
                            loss_type(weight=invalid_weight)
                        with self.assertRaises(ValueError):
                            loss_type(weight=[1.0, invalid_weight])
                        with self.assertRaises(ValueError):
                            loss_type(weight=torch.tensor([invalid_weight]))
                with self.assertRaises(ValueError):
                    loss_type(weight=[[1.0]])
                with self.assertRaises(ValueError):
                    loss_type(reduction="unsupported")
                with self.assertRaises(ValueError):
                    loss_type(weight=[1.0, 2.0, 3.0])(torch.ones(1, 2, 2), torch.ones(1, 2, 2))
                with self.assertRaises(AssertionError):
                    loss_type()(torch.ones(1, 2, 2), torch.ones(1, 1, 2))
                with self.assertRaises(ValueError):
                    loss_type()(torch.ones(1, 2), torch.ones(1, 2))
                with self.assertRaises(TypeError):
                    loss_type()(torch.ones(1, 2, 2, dtype=torch.int64), torch.ones(1, 2, 2))
        for invalid_empty_weight in (-1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid_empty_weight=invalid_empty_weight), self.assertRaises(ValueError):
                SoftL1ACELoss(empty_weight=invalid_empty_weight)

    def test_single_channel_warnings(self):
        prediction = torch.tensor([[[0.2, 0.8]]])
        target = torch.tensor([[[0.0, 1.0]]])
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loss_type(softmax=True, to_onehot_y=True, include_background=False)(prediction, target)
                self.assertEqual(len(caught), 3)

    def test_soft_empty_weight_masks_bins(self):
        prediction = torch.tensor([[[0.25, 0.75]]], requires_grad=True)
        target = torch.tensor([[[0.0, 1.0]]])
        result = SoftL1ACELoss(num_bins=2, empty_weight=1.1, ignore_empty_classes=False)(prediction, target)
        torch.testing.assert_close(result, torch.tensor(0.0))
        result.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction.grad))

    def test_soft_fractional_counts_are_normalized(self):
        prediction = torch.tensor([[[0.5]]])
        target = torch.tensor([[[1.0]]])
        result = SoftL1ACELoss(num_bins=2, empty_weight=0.0)(prediction, target)
        torch.testing.assert_close(result, torch.tensor(0.5))

    def test_dtype_and_gradcheck(self):
        target = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]], dtype=torch.float64)
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                prediction = torch.tensor([[[0.13, 0.31, 0.67, 0.89]]], dtype=torch.float64, requires_grad=True)
                result = loss_type(num_bins=5, ignore_empty_classes=False)(prediction, target)
                self.assertEqual(result.dtype, torch.float64)
                self.assertTrue(
                    torch.autograd.gradcheck(loss_type(num_bins=5, ignore_empty_classes=False), (prediction, target))
                )
                result.backward()
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertTrue(torch.any(prediction.grad != 0))

    def test_lower_precision_accumulates_in_float32(self):
        prediction = torch.tensor([[[0.1, 0.9]]], dtype=torch.float16)
        target = torch.tensor([[[0.0, 1.0]]], dtype=torch.float16)
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                result = loss_type(ignore_empty_classes=False)(prediction, target)
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.isfinite(result))

    def test_3d_spatial_input(self):
        prediction = torch.rand(2, 3, 4, 5, 6)
        labels = torch.randint(0, 3, (2, 1, 4, 5, 6))
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                result = loss_type(to_onehot_y=True, softmax=True)(prediction, labels)
                self.assertEqual(result.shape, torch.Size([]))
                self.assertTrue(torch.isfinite(result))

    def test_script_save(self):
        prediction = torch.rand(2, 3, 4, 5, 6)
        target = torch.nn.functional.one_hot(torch.randint(0, 3, (2, 4, 5, 6)), num_classes=3)
        target = target.movedim(-1, 1).float()
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                check_script_save(loss_type(num_bins=5), prediction, target)

    @skip_if_no_cuda
    def test_cuda_float32_and_float64(self):
        for dtype in (torch.float32, torch.float64):
            for loss_type in self.loss_types:
                with self.subTest(dtype=dtype, loss=loss_type.__name__):
                    prediction = torch.tensor([[[0.1, 0.9]]], device="cuda", dtype=dtype, requires_grad=True)
                    target = torch.tensor([[[0.0, 1.0]]], device="cuda", dtype=dtype)
                    result = loss_type(ignore_empty_classes=False).cuda()(prediction, target)
                    self.assertEqual(result.device.type, "cuda")
                    self.assertEqual(result.dtype, dtype)
                    result.backward()
                    self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_input_is_not_mutated(self):
        prediction = torch.tensor([[[0.1, 0.9]]])
        target = torch.tensor([[[0.0, 1.0]]])
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                prediction_before = prediction.clone()
                target_before = target.clone()
                weight = torch.tensor([2.0])
                loss = loss_type(weight=weight)
                loss(prediction, target)
                torch.testing.assert_close(prediction, prediction_before)
                torch.testing.assert_close(target, target_before)
                torch.testing.assert_close(loss.class_weight, weight)

    def test_target_has_no_gradient(self):
        for loss_type in self.loss_types:
            with self.subTest(loss=loss_type.__name__):
                prediction = torch.tensor([[[0.1, 0.9]]], requires_grad=True)
                target = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
                loss_type(ignore_empty_classes=False)(prediction, target).backward()
                self.assertIsNotNone(prediction.grad)
                self.assertIsNone(target.grad)


if __name__ == "__main__":
    unittest.main()
