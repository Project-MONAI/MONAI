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

from monai.metrics import ClDiceMetric, compute_cldice


class TestClDiceMetric(unittest.TestCase):
    """Unit tests for ClDiceMetric and compute_cldice across 2D and 3D inputs."""

    def test_2d_and_3d_modes(self) -> None:
        """Test partial-overlap 2D and 3D masks with distinct scores per mode and background option."""
        y_2d = torch.zeros((2, 2, 16, 16))
        y_pred_2d = torch.zeros((2, 2, 16, 16))
        y_2d[:, 0, 2, 2:6] = 1.0
        y_pred_2d[:, 0, 2, 2:5] = 1.0
        y_pred_2d[:, 0, 12, 2:3] = 1.0

        y_2d[:, 1, 5, 2:7] = 1.0
        y_2d[:, 1, 10, 2:7] = 1.0
        y_pred_2d[:, 1, 5, 2:10] = 1.0

        y_3d = torch.zeros((2, 2, 12, 12, 12))
        y_pred_3d = torch.zeros((2, 2, 12, 12, 12))
        y_3d[:, 0, 2, 2, 2:6] = 1.0
        y_pred_3d[:, 0, 2, 2, 2:5] = 1.0
        y_pred_3d[:, 0, 8, 2, 2:3] = 1.0

        y_3d[:, 1, 5, 2, 2:7] = 1.0
        y_3d[:, 1, 9, 2, 2:7] = 1.0
        y_pred_3d[:, 1, 5, 2, 2:10] = 1.0

        cases = [
            (y_pred_2d, y_2d, "tprec", False, 0.625),
            (y_pred_2d, y_2d, "tsens", False, 0.500),
            (y_pred_2d, y_2d, "cldice", False, 5.0 / 9.0),
            (y_pred_2d, y_2d, "tprec", True, (0.75 + 0.625) / 2.0),
            (y_pred_2d, y_2d, "tsens", True, (0.75 + 0.500) / 2.0),
            (y_pred_2d, y_2d, "cldice", True, (0.75 + 5.0 / 9.0) / 2.0),
            (y_pred_3d, y_3d, "cldice", False, 5.0 / 9.0),
            (y_pred_3d, y_3d, "cldice", True, (0.75 + 5.0 / 9.0) / 2.0),
        ]
        for y_pred, y, mode, include_bg, expected in cases:
            with self.subTest(ndim=y_pred.ndim, mode=mode, include_bg=include_bg):
                metric = ClDiceMetric(include_background=include_bg, mode=mode, reduction="mean")
                metric(y_pred, y)
                result = metric.aggregate()
                self.assertAlmostEqual(result.item(), expected, places=4)

    def test_solid_squares_zero_skeleton_fallback(self) -> None:
        """Verify that disjoint 10x10 squares score 0.0 while identical 10x10 squares score 1.0 at iter_=3."""
        y_pred_disjoint = torch.zeros((1, 1, 40, 40))
        y_ref = torch.zeros((1, 1, 40, 40))
        y_pred_disjoint[..., 5:15, 5:15] = 1.0
        y_ref[..., 25:35, 25:35] = 1.0

        for mode in ("cldice", "tprec", "tsens"):
            disjoint_score = compute_cldice(y_pred_disjoint, y_ref, include_background=True, iter_=3, mode=mode)
            identical_score = compute_cldice(y_ref, y_ref, include_background=True, iter_=3, mode=mode)
            self.assertEqual(disjoint_score.item(), 0.0)
            self.assertEqual(identical_score.item(), 1.0)

    def test_empty_masks(self) -> None:
        """Verify that two empty masks return 1.0 and empty reference with non-empty prediction returns NaN."""
        y_pred = torch.zeros((1, 2, 16, 16))
        y = torch.zeros((1, 2, 16, 16))
        y_pred[0, 1, 4:8, 4:8] = 1.0

        res = compute_cldice(y_pred, y, include_background=True)
        self.assertEqual(res[0, 0].item(), 1.0)
        self.assertTrue(torch.isnan(res[0, 1]))


if __name__ == "__main__":
    unittest.main()
