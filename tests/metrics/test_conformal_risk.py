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

from monai.metrics import (
    ConformalRiskCalibrator,
    ConformalRiskPredictor,
    Coverage,
    SetSize,
    compute_coverage,
    compute_set_size,
)
from monai.metrics.conformal_risk import false_negative_loss, miscoverage_loss
from tests.test_utils import assert_allclose


class TestMiscoverageLoss(unittest.TestCase):
    def test_full_coverage_zero_loss(self):
        # 2 images, 3 classes, 2 voxels. True class 0 always included.
        sets = torch.ones(2, 3, 2, dtype=torch.bool)
        labels = torch.zeros(2, 1, 2, dtype=torch.long)
        loss = miscoverage_loss(sets, labels)
        assert_allclose(loss, torch.zeros(2), atol=1e-6)

    def test_no_coverage_unit_loss(self):
        sets = torch.zeros(2, 3, 2, dtype=torch.bool)  # empty sets
        labels = torch.zeros(2, 1, 2, dtype=torch.long)
        loss = miscoverage_loss(sets, labels)
        assert_allclose(loss, torch.ones(2), atol=1e-6)

    def test_half_coverage(self):
        # image 0: voxel 0 covered (class 0 in set), voxel 1 not -> loss 0.5
        sets = torch.tensor([[[True, False], [False, False], [False, False]]])  # (1, 3, 2)
        labels = torch.zeros(1, 1, 2, dtype=torch.long)
        loss = miscoverage_loss(sets, labels)
        assert_allclose(loss, torch.tensor([0.5]), atol=1e-6)


class TestFalseNegativeLoss(unittest.TestCase):
    def test_ignores_background(self):
        # image: voxel 0 bg (label 0), voxel 1 fg class 1 not covered -> FNR 1.0
        sets = torch.zeros(1, 3, 2, dtype=torch.bool)
        labels = torch.tensor([[[0, 1]]])
        loss = false_negative_loss(sets, labels)
        assert_allclose(loss, torch.tensor([1.0]), atol=1e-6)

    def test_no_foreground_zero_loss(self):
        # all background -> denom 0 -> loss 0 by convention
        sets = torch.zeros(1, 3, 2, dtype=torch.bool)
        labels = torch.zeros(1, 1, 2, dtype=torch.long)
        loss = false_negative_loss(sets, labels)
        assert_allclose(loss, torch.tensor([0.0]), atol=1e-6)


class TestConformalRiskCalibrator(unittest.TestCase):
    def test_calibrate_miscoverage_monotone(self):
        # 4 images, 2 classes, 2 voxels each. True class 0; softmax[0] gives true-prob
        # 0.55, 0.65, 0.75, 0.85 -> scores 0.45, 0.35, 0.25, 0.15. With grid step 0.1:
        # at lam=0.2 only the 0.15 image is covered -> risk 0.75; at lam=0.3 the
        # 0.25 and 0.15 images are covered -> risk 0.5; at lam=0.4 three covered -> risk 0.25.
        # Smallest grid point with risk <= alpha_eff (0.25, see below) is 0.4.
        p = [0.55, 0.65, 0.75, 0.85]
        probs = torch.stack([torch.tensor([[pi, pi], [1 - pi, 1 - pi]]) for pi in p])  # (4, C=2, spatial=2)
        labels = torch.zeros(4, 1, 2, dtype=torch.long)
        # n=4, B=1: alpha_eff = (5*0.4 - 1)/4 = 0.25; emp_risk hits 0.25 first at lam=0.4.
        cal = ConformalRiskCalibrator(alpha=0.4, loss="miscoverage", lam_grid=torch.linspace(0.0, 1.0, 11))  # step 0.1
        cal.accumulate(probs, labels)
        lam = cal.calibrate()
        assert_allclose(lam, torch.tensor(0.4), atol=1e-6)

    def test_calibrate_returns_scalar_in_grid(self):
        probs = torch.rand(5, 3, 2, 2)
        probs = probs / probs.sum(dim=1, keepdim=True)
        labels = torch.randint(0, 3, (5, 1, 2, 2))
        cal = ConformalRiskCalibrator(alpha=0.1)
        cal.accumulate(probs, labels)
        lam = cal.calibrate()
        self.assertEqual(lam.ndim, 0)
        self.assertTrue(0.0 <= lam.item() <= 1.0)

    def test_calibrate_multi_batch(self):
        # two batches each with 2 images; same probs as the monotone test split in half
        probs_a = torch.stack([torch.tensor([[0.55, 0.55], [0.45, 0.45]]), torch.tensor([[0.65, 0.65], [0.35, 0.35]])])
        probs_b = torch.stack([torch.tensor([[0.75, 0.75], [0.25, 0.25]]), torch.tensor([[0.85, 0.85], [0.15, 0.15]])])
        labels = torch.zeros(2, 1, 2, dtype=torch.long)
        cal = ConformalRiskCalibrator(alpha=0.4, lam_grid=torch.linspace(0.0, 1.0, 11))
        cal.accumulate(probs_a, labels)
        cal.accumulate(probs_b, labels)
        lam = cal.calibrate()
        assert_allclose(lam, torch.tensor(0.4), atol=1e-6)

    def test_include_background_drops_bg_voxels(self):
        # 1 image, 3 classes, 2 voxels. voxel 0 bg (label 0) has a HARD true class (score 0.5);
        # voxel 1 fg class 1 is easy (score 0.2). n=1, alpha=0.5 -> alpha_eff = (2*0.5-1)/1 = 0,
        # so we need risk exactly 0. Dropping the bg voxel lets lam = 0.2; keeping it forces
        # lam = 0.5 to also cover the hard bg voxel -> the flag genuinely changes the result.
        probs = torch.tensor([[[0.5, 0.1], [0.3, 0.8], [0.2, 0.1]]])  # (1, 3, 2)
        labels = torch.tensor([[[0, 1]]])  # (1, 1, 2)
        grid = torch.linspace(0.0, 1.0, 11)
        lam_drop = ConformalRiskCalibrator(alpha=0.5, loss="miscoverage", include_background=False, lam_grid=grid)
        lam_drop.accumulate(probs, labels)
        assert_allclose(lam_drop.calibrate(), torch.tensor(0.2), atol=1e-6)
        lam_keep = ConformalRiskCalibrator(alpha=0.5, loss="miscoverage", include_background=True, lam_grid=grid)
        lam_keep.accumulate(probs, labels)
        assert_allclose(lam_keep.calibrate(), torch.tensor(0.5), atol=1e-6)

    def test_variable_size_volumes(self):
        # Calibration images with DIFFERENT spatial sizes (4 vs 9 voxels) must work — each image
        # is stored separately. label 0 everywhere; class-0 scores: imgs 0.3, 0.4 (batch1), 0.1
        # (batch2). n=3, alpha=0.5 -> alpha_eff = (4*0.5-1)/3 = 1/3; emp_risk first <= 1/3 at lam=0.3.
        def img(p0, n):  # one image (C=2, n voxels) with class-0 prob p0
            return torch.stack([torch.full((n,), p0), torch.full((n,), 1 - p0)])

        probs_1 = torch.stack([img(0.7, 4), img(0.6, 4)])  # (2, 2, 4) -> class-0 scores 0.3, 0.4
        probs_2 = img(0.9, 9).unsqueeze(0)  # (1, 2, 9) -> class-0 score 0.1
        cal = ConformalRiskCalibrator(alpha=0.5, loss="miscoverage", lam_grid=torch.linspace(0.0, 1.0, 11))
        cal.accumulate(probs_1, torch.zeros(2, 1, 4, dtype=torch.long))
        cal.accumulate(probs_2, torch.zeros(1, 1, 9, dtype=torch.long))
        lam = cal.calibrate()
        assert_allclose(lam, torch.tensor(0.3), atol=1e-6)

    def test_false_negative_loss_calibrate(self):
        # 2 images, 3 classes, 2 voxels. img0 bg-only, img1 fg class 1 prob 0.65 -> score 0.35.
        # n=2, alpha=0.5 -> alpha_eff = (3*0.5-1)/2 = 0.25. emp_risk = 0.5*1{0.35 > lam}, so it
        # clears 0.25 first at lam=0.4 (img1's fg voxel covered).
        probs = torch.tensor(
            [
                [[0.9, 0.9], [0.05, 0.05], [0.05, 0.05]],  # img0: both voxels bg-dominant
                [[0.25, 0.25], [0.65, 0.65], [0.1, 0.1]],  # img1: both voxels class 1 prob 0.65
            ]
        )  # (2, 3, 2)
        labels = torch.tensor([[[0, 0]], [[1, 1]]])  # (2, 1, 2)
        cal = ConformalRiskCalibrator(alpha=0.5, loss="false_negative", lam_grid=torch.linspace(0.0, 1.0, 11))
        cal.accumulate(probs, labels)
        lam = cal.calibrate()
        assert_allclose(lam, torch.tensor(0.4), atol=1e-6)

    def test_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            ConformalRiskCalibrator(alpha=0.0)
        with self.assertRaises(ValueError):
            ConformalRiskCalibrator(alpha=1.0)

    def test_rejects_unknown_loss(self):
        with self.assertRaises(ValueError):
            ConformalRiskCalibrator(loss="brier")

    def test_rejects_bad_lam_grid(self):
        with self.assertRaises(ValueError):
            ConformalRiskCalibrator(lam_grid=torch.tensor([[-0.1], [0.5]]))

    def test_calibrate_empty_raises(self):
        with self.assertRaises(RuntimeError):
            ConformalRiskCalibrator().calibrate()

    def test_reset_clears_buffers(self):
        cal = ConformalRiskCalibrator()
        # probs (1, 2, 1): 1 image, 2 classes, 1 voxel; labels (1, 1, 1)
        cal.accumulate(torch.tensor([[[0.5], [0.5]]]), torch.zeros(1, 1, 1, dtype=torch.long))
        cal.reset()
        with self.assertRaises(RuntimeError):
            cal.calibrate()


class TestConformalRiskPredictor(unittest.TestCase):
    def test_sets_and_mask(self):
        # 1 image, 3 classes, 3 voxels.
        # voxel 0: probs [0.9, 0.05, 0.05] -> scores [0.1, 0.95, 0.95] -> only class 0 in set
        # voxel 1: probs [0.5, 0.3, 0.2]  -> scores [0.5, 0.7, 0.8]   -> only class 0 in set (lam=0.6)
        # voxel 2: probs [0.4, 0.4, 0.2]  -> scores [0.6, 0.6, 0.8]   -> classes 0,1 in set (ambiguous)
        probs = torch.tensor([[[0.9, 0.5, 0.4], [0.05, 0.3, 0.4], [0.05, 0.2, 0.2]]])  # (1, 3, 3): B=1, C=3, spatial=3
        lam = torch.tensor(0.6)
        predictor = ConformalRiskPredictor(lam=lam)
        sets, mask, probs_out = predictor(probs)
        self.assertEqual(sets.shape, (1, 3, 3))
        self.assertEqual(mask.shape, (1, 1, 3))
        self.assertEqual(mask.dtype, torch.bool)
        # voxel 0: only class 0 -> not ambiguous
        self.assertFalse(mask[0, 0, 0].item())
        # voxel 1: only class 0 -> not ambiguous
        self.assertFalse(mask[0, 0, 1].item())
        # voxel 2: classes 0 and 1 -> ambiguous
        self.assertTrue(mask[0, 0, 2].item())
        # sets correct
        self.assertTrue(sets[0, 0].all())  # class 0 always in set
        self.assertFalse(sets[0, 1, 0].item())
        self.assertTrue(sets[0, 1, 2].item())  # class 1 in set for voxel 2
        self.assertFalse(sets[0, 2].any().item())  # class 2 never in set

    def test_include_background_zeroes_bg_argmax_mask(self):
        # 1 image, 3 classes, 2 voxels. voxel 0 argmax 0 (bg), set ambiguous; with
        # include_background=False mask should be False there.
        # voxel 0: probs [0.4, 0.3, 0.3] -> scores [0.6, 0.7, 0.7] -> set {0,1,2} at lam=0.75
        # voxel 1: probs [0.1, 0.8, 0.1] -> scores [0.9, 0.2, 0.9] -> set {1} only -> not ambiguous
        probs = torch.tensor([[[0.4, 0.1], [0.3, 0.8], [0.3, 0.1]]])  # (1, 3, 2)
        lam = torch.tensor(0.75)
        predictor = ConformalRiskPredictor(lam=lam, include_background=False)
        _, mask, _ = predictor(probs)
        # voxel 0: argmax==0 (bg) -> masked out despite ambiguous set
        self.assertFalse(mask[0, 0, 0].item())
        # voxel 1: only class 1 in set -> not ambiguous
        self.assertFalse(mask[0, 0, 1].item())

    def test_set_threshold(self):
        predictor = ConformalRiskPredictor(lam=torch.tensor(0.5))
        predictor.set_threshold(torch.tensor(0.9))
        assert_allclose(predictor.lam, torch.tensor(0.9), atol=1e-6)

    def test_rejects_non_tensor_lam(self):
        with self.assertRaises(TypeError):
            ConformalRiskPredictor(lam=0.5)

    def test_rejects_bad_probs_shape(self):
        predictor = ConformalRiskPredictor(lam=torch.tensor(0.5))
        with self.assertRaises(ValueError):
            predictor(torch.zeros(5))  # 1-D


class TestCoverageMetric(unittest.TestCase):
    def test_compute_coverage_full(self):
        sets = torch.ones(2, 3, 4, dtype=torch.bool)
        labels = torch.zeros(2, 1, 4, dtype=torch.long)
        cov = compute_coverage(sets, labels)
        assert_allclose(cov, torch.ones(2), atol=1e-6)

    def test_compute_coverage_partial(self):
        # image 0: voxel 0 covered (class 0 in set), voxel 1 not -> cov 0.5
        sets = torch.tensor([[[True, False], [False, False], [False, False]]])
        labels = torch.zeros(1, 1, 2, dtype=torch.long)
        cov = compute_coverage(sets, labels)
        assert_allclose(cov, torch.tensor([0.5]), atol=1e-6)

    def test_coverage_aggregate_mean(self):
        metric = Coverage()
        sets = torch.ones(2, 3, 4, dtype=torch.bool)
        labels = torch.zeros(2, 1, 4, dtype=torch.long)
        metric(sets, labels)
        result = metric.aggregate()
        assert_allclose(result, torch.tensor(1.0), atol=1e-6)

    def test_coverage_aggregate_get_not_nans(self):
        metric = Coverage(get_not_nans=True)
        sets = torch.ones(1, 2, 2, dtype=torch.bool)
        labels = torch.zeros(1, 1, 2, dtype=torch.long)
        metric(sets, labels)
        result, not_nans = metric.aggregate()
        self.assertEqual(not_nans.item(), 1.0)
        assert_allclose(result, torch.tensor(1.0), atol=1e-6)


class TestSetSizeMetric(unittest.TestCase):
    def test_compute_set_size(self):
        # 2 images, 3 classes, 2 voxels.
        # img0: voxel0 {0,1} size 2, voxel1 {0} size 1 -> mean 1.5
        # img1: voxel0 {0,1,2} size 3, voxel1 {0,1} size 2 -> mean 2.5
        sets = torch.tensor(
            [
                [[True, True], [True, False], [False, False]],  # img0: voxel0 {0,1}, voxel1 {0}
                [[True, True], [True, True], [True, False]],  # img1: voxel0 {0,1,2}, voxel1 {0,1}
            ]
        )  # (2, 3, 2)
        sizes = compute_set_size(sets)
        assert_allclose(sizes, torch.tensor([1.5, 2.5]), atol=1e-6)

    def test_setsize_aggregate_mean(self):
        metric = SetSize()
        sets = torch.ones(2, 3, 4, dtype=torch.bool)
        metric(sets, None)
        result = metric.aggregate()
        assert_allclose(result, torch.tensor(3.0), atol=1e-6)

    def test_setsize_rejects_non_tensor(self):
        metric = SetSize()
        with self.assertRaises(TypeError):
            metric._compute_tensor([[True, True]], None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
