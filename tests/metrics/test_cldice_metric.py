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
    def test_2d_and_3d_modes(self):
        cases = [
            ((2, 2, 16, 16), "cldice", True),
            ((2, 2, 16, 16), "tprec", False),
            ((2, 2, 16, 16), "tsens", True),
            ((2, 2, 12, 12, 12), "cldice", False),
        ]
        for shape, mode, include_bg in cases:
            with self.subTest(shape=shape, mode=mode, include_bg=include_bg):
                y_pred = torch.ones(shape)
                y = torch.ones(shape)
                metric = ClDiceMetric(include_background=include_bg, mode=mode, reduction="mean")
                metric(y_pred, y)
                result = metric.aggregate()
                self.assertAlmostEqual(result.item(), 1.0, places=4)

    def test_empty_masks(self):
        y_pred = torch.zeros((1, 2, 16, 16))
        y = torch.zeros((1, 2, 16, 16))
        y_pred[0, 1, 4:8, 4:8] = 1.0

        res = compute_cldice(y_pred, y, include_background=True)
        self.assertEqual(res[0, 0].item(), 1.0)
        self.assertTrue(torch.isnan(res[0, 1]))


if __name__ == "__main__":
    unittest.main()
