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

from monai.inferers import ConformalCalibrator, ConformalPredictor
from monai.inferers.conformal_predictor import _quantile_threshold
from tests.test_utils import assert_allclose


class TestQuantileThreshold(unittest.TestCase):
    def test_exact_coverage_rank(self):
        # alpha=0.1, n=9 -> rank = ceil(10*0.9) = 9 -> kth=9 -> scores[8]
        scores = torch.arange(1, 10, dtype=torch.float32)  # 1..9
        qhat = _quantile_threshold(scores, alpha=0.1)
        assert_allclose(qhat, torch.tensor(9.0))

    def test_small_n_returns_inf(self):
        # n too small for alpha (ceil((n+1)(1-alpha)) > n): no finite score gives the
        # guarantee, so qhat must be +inf (full prediction set), not the max score.
        scores = torch.linspace(0.0, 1.0, 5)
        qhat = _quantile_threshold(scores, alpha=1e-6)
        self.assertTrue(torch.isinf(qhat))
        # n=8, alpha=0.1 -> rank ceil(9*0.9)=9 > 8 -> inf
        self.assertTrue(torch.isinf(_quantile_threshold(torch.rand(8), alpha=0.1)))
        # n=9, alpha=0.1 -> rank 9 <= 9 -> finite
        self.assertFalse(torch.isinf(_quantile_threshold(torch.rand(9), alpha=0.1)))

    def test_marginal_coverage_monte_carlo(self):
        # Canonical conformal regression test: with exchangeable scores, held-out coverage
        # P(score_test <= qhat) must be >= 1 - alpha in expectation over calibration draws.
        g = torch.Generator().manual_seed(0)
        alpha, n, n_test, trials = 0.2, 20, 50, 400
        covered = 0
        total = 0
        for _ in range(trials):
            cal_scores = torch.rand(n, generator=g)
            test_scores = torch.rand(n_test, generator=g)
            qhat = _quantile_threshold(cal_scores, alpha=alpha)
            covered += int((test_scores <= qhat).sum())
            total += n_test
        coverage = covered / total
        # target 0.8; allow Monte-Carlo slack below and the standard (1-alpha)+1/(n+1) above
        self.assertGreaterEqual(coverage, 1 - alpha - 0.02)
        self.assertLessEqual(coverage, 1 - alpha + 1 / (n + 1) + 0.02)

    def test_marginal_coverage_small_n(self):
        # in the small-n regime the +inf fallback must yield full coverage, not under-cover
        g = torch.Generator().manual_seed(0)
        for _ in range(50):
            qhat = _quantile_threshold(torch.rand(5, generator=g), alpha=0.1)
            self.assertTrue((torch.rand(20, generator=g) <= qhat).all())

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            _quantile_threshold(torch.empty(0), alpha=0.1)

    def test_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            _quantile_threshold(torch.ones(4), alpha=0.0)
        with self.assertRaises(ValueError):
            _quantile_threshold(torch.ones(4), alpha=1.0)


class TestConformalCalibrator(unittest.TestCase):
    def _cal_batch(self, probs, labels, include_background=True, alpha=0.1):
        cal = ConformalCalibrator(alpha=alpha, include_background=include_background)
        cal.accumulate(probs, labels)
        return cal.calibrate()

    def test_classification_single_batch(self):
        # 3 classes, 10 samples; true class prob 0.8 each -> score 0.2 each
        probs = torch.full((10, 3), 0.1)
        probs[:, 0] = 0.8
        probs = probs / probs.sum(dim=1, keepdim=True)
        labels = torch.zeros(10, dtype=torch.long)
        qhat = self._cal_batch(probs, labels)
        # all scores == 0.2 -> qhat == 0.2
        assert_allclose(qhat, torch.tensor(0.2), atol=1e-6)

    def test_classification_multi_batch(self):
        cal = ConformalCalibrator(alpha=0.1)
        # batch 1: 5 samples, true prob 0.9 -> score 0.1
        probs1 = torch.full((5, 2), 0.1)
        probs1[:, 0] = 0.9
        probs1 = probs1 / probs1.sum(dim=1, keepdim=True)
        cal.accumulate(probs1, torch.zeros(5, dtype=torch.long))
        # batch 2: 5 samples, true prob 0.5 -> score 0.5
        probs2 = torch.full((5, 2), 0.5)
        probs2 = probs2 / probs2.sum(dim=1, keepdim=True)
        cal.accumulate(probs2, torch.zeros(5, dtype=torch.long))
        # combined 10 scores: five 0.1, five 0.5; rank=ceil(11*0.9)=10 -> kth=10 -> max
        qhat = cal.calibrate()
        assert_allclose(qhat, torch.tensor(0.5), atol=1e-6)

    def test_segmentation_voxel_reshape(self):
        # (B=2, C=3, H=2, W=3) -> n=12 voxels with constant true-prob 0.8 -> score 0.2 everywhere
        probs = torch.zeros(2, 3, 2, 3)
        probs[:, 0] = 0.8
        probs[:, 1] = 0.1
        probs[:, 2] = 0.1  # already normalized: 0.8 + 0.1 + 0.1 = 1
        labels = torch.zeros(2, 1, 2, 3, dtype=torch.long)
        qhat = self._cal_batch(probs, labels)
        assert_allclose(qhat, torch.tensor(0.2), atol=1e-6)

    def test_exclude_background_scores_foreground_only(self):
        # foreground voxel (class 1): full softmax kept, score = 1 - softmax[1]
        probs = torch.tensor([[[0.8], [0.1], [0.1]]])  # (B=1, C=3, spatial=1)
        labels = torch.tensor([[[1]]])  # (1,1,1) class 1
        # alpha=0.5 keeps rank feasible for n=1 (small-n now yields +inf by design)
        qhat = self._cal_batch(probs, labels, include_background=False, alpha=0.5)
        # true class-1 prob 0.1 -> score 0.9
        assert_allclose(qhat, torch.tensor(0.9), atol=1e-6)

    def test_exclude_background_drops_bg_voxels(self):
        # one bg voxel (class 0) + one fg voxel (class 2); bg must be excluded so only the
        # fg score (1 - 0.7 = 0.3) calibrates the threshold, not bg's tiny score.
        probs = torch.tensor([[[0.9, 0.2], [0.05, 0.1], [0.05, 0.7]]])  # (B=1, C=3, spatial=2)
        labels = torch.tensor([[[0, 2]]])  # (1,1,2): voxel0 bg, voxel1 class 2
        qhat = self._cal_batch(probs, labels, include_background=False, alpha=0.5)
        assert_allclose(qhat, torch.tensor(0.3), atol=1e-6)

    def test_unsupported_score_raises(self):
        with self.assertRaises(ValueError):
            ConformalCalibrator(score="aps")

    def test_calibrate_empty_raises(self):
        with self.assertRaises(RuntimeError):
            ConformalCalibrator().calibrate()

    def test_invalid_labels_are_dropped_not_clamped(self):
        # labels -1 and 99 (out of range for C=3) must be dropped, not clamped to 0/2.
        # If clamped, score for "label 99" would be 1 - softmax[2] = 0.9, corrupting qhat.
        probs = torch.tensor([[[0.8, 0.8], [0.1, 0.1], [0.1, 0.1]]])  # (1, 3, 2)
        labels = torch.tensor([[[-1, 99]]])  # both invalid
        cal = ConformalCalibrator(alpha=0.1)
        cal.accumulate(probs, labels)
        with self.assertRaises(RuntimeError):
            # all labels dropped -> no scores -> calibrate raises
            cal.calibrate()

    def test_mixed_valid_invalid_labels_keeps_valid(self):
        # one valid label (class 0, score 0.2) + one invalid (class 99, dropped).
        # qhat should reflect only the valid score.
        probs = torch.tensor([[[0.8, 0.8], [0.1, 0.1], [0.1, 0.1]]])  # (1, 3, 2)
        labels = torch.tensor([[[0, 99]]])  # voxel0 valid, voxel1 invalid
        qhat = self._cal_batch(probs, labels, alpha=0.5)
        assert_allclose(qhat, torch.tensor(0.2), atol=1e-6)


class TestConformalPredictor(unittest.TestCase):
    def test_set_and_predict(self):
        qhat = torch.tensor(0.3)
        inferer = ConformalPredictor(qhat=qhat)

        def net(x):
            # logits: (1, 3, 2, 2) where channel 0 dominates
            logits = torch.zeros(1, 3, 2, 2)
            logits[:, 0] = 5.0
            logits[:, 1] = 0.0
            logits[:, 2] = 0.0
            return logits

        sets = inferer(torch.zeros(1, 1, 2, 2), net)
        # softmax[0] ~ 0.99 -> score 0.01 <= 0.3 -> True; others score ~0.99 -> False
        self.assertEqual(sets.shape, (1, 3, 2, 2))
        self.assertEqual(sets.dtype, torch.bool)
        self.assertTrue(sets[:, 0].all())
        self.assertFalse(sets[:, 1].any())
        self.assertFalse(sets[:, 2].any())

    def test_no_threshold_raises(self):
        inferer = ConformalPredictor()  # qhat None

        def net(x):
            return torch.zeros(1, 2)

        with self.assertRaises(RuntimeError):
            inferer(torch.zeros(1, 1), net)

    def test_calibrate_then_predict(self):
        # tiny deterministic "model": always predict [0.9, 0.1]
        class Net(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 2)
                logits[:, 0] = 2.1972  # log(0.9 / 0.1)
                return logits

        net = Net()
        # calibration set: 10 samples, true label always 0
        cal_loader = [{"image": torch.zeros(2, 1), "label": torch.zeros(2, 1, dtype=torch.long)} for _ in range(5)]

        inferer = ConformalPredictor(alpha=0.1)
        qhat = inferer.calibrate(net, cal_loader, device=torch.device("cpu"))
        # true prob 0.9 -> score 0.1 for all 10 samples; rank=ceil(11*0.9)=10 -> max score 0.1
        assert_allclose(qhat, torch.tensor(0.1), atol=1e-4)
        sets = inferer(torch.zeros(1, 1), net)
        self.assertTrue(sets[0, 0].item())
        self.assertFalse(sets[0, 1].item())

    def test_bad_network_output_raises(self):
        inferer = ConformalPredictor(qhat=torch.tensor(0.5))

        def net(x):
            return [0.0, 0.0]  # not a tensor

        with self.assertRaises(TypeError):
            inferer(torch.zeros(1, 1), net)

    def test_set_threshold_rejects_non_scalar(self):
        inferer = ConformalPredictor()
        with self.assertRaises(ValueError):
            inferer.set_threshold(torch.tensor([0.1, 0.2]))

    def test_set_threshold_rejects_non_tensor(self):
        inferer = ConformalPredictor()
        with self.assertRaises(TypeError):
            inferer.set_threshold(0.5)  # type: ignore[arg-type]

    def test_calibrate_restores_training_state(self):
        class Net(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 2)
                logits[:, 0] = 2.1972
                return logits

        net = Net()
        net.train()
        self.assertTrue(net.training)
        cal_loader = [{"image": torch.zeros(2, 1), "label": torch.zeros(2, 1, dtype=torch.long)} for _ in range(5)]
        inferer = ConformalPredictor(alpha=0.1)
        inferer.calibrate(net, cal_loader, device=torch.device("cpu"))
        # training state must be restored to True
        self.assertTrue(net.training)

    def test_calibrate_preserves_eval_state(self):
        class Net(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 2)
                logits[:, 0] = 2.1972
                return logits

        net = Net()
        net.eval()
        self.assertFalse(net.training)
        cal_loader = [{"image": torch.zeros(2, 1), "label": torch.zeros(2, 1, dtype=torch.long)} for _ in range(5)]
        inferer = ConformalPredictor(alpha=0.1)
        inferer.calibrate(net, cal_loader, device=torch.device("cpu"))
        self.assertFalse(net.training)


if __name__ == "__main__":
    unittest.main()
