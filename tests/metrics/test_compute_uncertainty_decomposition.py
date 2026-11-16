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

import math
import unittest

import torch

from monai.metrics import UncertaintyDecompositionMetric, compute_uncertainty_decomposition

LN2 = math.log(2)


class TestComputeUncertaintyDecomposition(unittest.TestCase):

    def test_identical_confident_samples_have_no_uncertainty(self):
        y_pred = torch.zeros(4, 2, 3, 3)
        y_pred[:, 0] = 1.0
        result = compute_uncertainty_decomposition(y_pred, spatial_map=True)
        for key in ("total", "aleatoric", "epistemic"):
            self.assertEqual(result[key].shape, (3, 3))
            torch.testing.assert_close(result[key], torch.zeros(3, 3), atol=1e-6, rtol=0)

    def test_disagreeing_confident_samples_are_epistemic(self):
        y_pred = torch.zeros(2, 2, 4)
        y_pred[0, 0] = 1.0
        y_pred[1, 1] = 1.0
        result = compute_uncertainty_decomposition(y_pred, spatial_map=True)
        torch.testing.assert_close(result["total"], torch.full((4,), LN2), atol=1e-5, rtol=0)
        torch.testing.assert_close(result["aleatoric"], torch.zeros(4), atol=1e-5, rtol=0)
        torch.testing.assert_close(result["epistemic"], torch.full((4,), LN2), atol=1e-5, rtol=0)

    def test_agreeing_uniform_samples_are_aleatoric(self):
        y_pred = torch.full((5, 2, 4), 0.5)
        result = compute_uncertainty_decomposition(y_pred, spatial_map=True)
        torch.testing.assert_close(result["total"], torch.full((4,), LN2), atol=1e-5, rtol=0)
        torch.testing.assert_close(result["aleatoric"], torch.full((4,), LN2), atol=1e-5, rtol=0)
        torch.testing.assert_close(result["epistemic"], torch.zeros(4), atol=1e-5, rtol=0)

    def test_scalar_reductions_and_total_is_sum_of_parts(self):
        y_pred = torch.softmax(torch.randn(6, 3, 5, 5, generator=torch.Generator().manual_seed(0)), dim=1)
        maps = compute_uncertainty_decomposition(y_pred, spatial_map=True)
        torch.testing.assert_close(maps["total"], maps["aleatoric"] + maps["epistemic"], atol=1e-5, rtol=0)
        self.assertTrue(bool((maps["epistemic"] >= 0).all()))
        means = compute_uncertainty_decomposition(y_pred, scalar_reduction="mean")
        sums = compute_uncertainty_decomposition(y_pred, scalar_reduction="sum")
        for key in maps:
            torch.testing.assert_close(means[key], maps[key].mean())
            torch.testing.assert_close(sums[key], maps[key].sum())

    def test_metric_class_matches_function(self):
        y_pred = torch.softmax(torch.randn(4, 2, 3, 3, 3, generator=torch.Generator().manual_seed(1)), dim=1)
        expected = compute_uncertainty_decomposition(y_pred, spatial_map=True)
        result = UncertaintyDecompositionMetric(spatial_map=True)(y_pred)
        for key in expected:
            torch.testing.assert_close(result[key], expected[key])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_uncertainty_decomposition(torch.rand(4))
        with self.assertRaises(ValueError):
            compute_uncertainty_decomposition(torch.rand(4, 2), scalar_reduction="max")


if __name__ == "__main__":
    unittest.main()
