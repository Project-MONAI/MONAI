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

from monai.losses.image_dissimilarity import LocalNormalizedCrossCorrelationLoss, make_gaussian_kernel

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    [
        {"spatial_dims": 1, "kernel_type": "rectangular", "reduction": "sum"},
        {
            "pred": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        -1.0 * 3,
    ],
    [
        {"spatial_dims": 1, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
    [
        {"spatial_dims": 1, "kernel_type": "triangular", "smooth_dr": 0.1},
        {
            "pred": torch.zeros(1, 2, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.zeros(1, 2, 3).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        0.0,
    ],
    [
        {"spatial_dims": 2, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3).reshape(1, 1, -1, 1).expand(1, 1, 3, 3).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3).reshape(1, 1, -1, 1).expand(1, 1, 3, 3).to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 1, 3, 3, 3)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 1, 3, 3, 3)
            .to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "rectangular"},
        {
            "pred": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device)
            ** 2,
        },
        -0.95801723,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "triangular", "kernel_size": 5},
        {
            "pred": torch.arange(0, 5)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 5, 5, 5)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 5)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 5, 5, 5)
            .to(dtype=torch.float, device=device)
            ** 2,
        },
        -0.918672,
    ],
    [
        {"spatial_dims": 3, "kernel_type": "gaussian"},
        {
            "pred": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device),
            "target": torch.arange(0, 3)
            .reshape(1, 1, -1, 1, 1)
            .expand(1, 3, 3, 3, 3)
            .to(dtype=torch.float, device=device)
            ** 2,
        },
        -0.95406944,
    ],
    # Regression tests for gh-8780: gaussian kernel_size > 3 was broken due to
    # truncated parameter being passed as pixel radius instead of sigma multiplier.
    # Identical images must yield loss == -1.0 for any kernel size.
    [
        {"spatial_dims": 1, "kernel_type": "gaussian", "kernel_size": 5},
        {
            "pred": torch.arange(0, 5).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 5).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
    [
        {"spatial_dims": 1, "kernel_type": "gaussian", "kernel_size": 9},
        {
            "pred": torch.arange(0, 9).reshape(1, 1, -1).to(dtype=torch.float, device=device),
            "target": torch.arange(0, 9).reshape(1, 1, -1).to(dtype=torch.float, device=device),
        },
        -1.0,
    ],
]


class TestLocalNormalizedCrossCorrelationLoss(unittest.TestCase):

    @parameterized.expand([1e-5, 1e-6])
    def test_low_variance_range(self, smooth_dr):
        # gh-9178: cancellation in float32 moments can produce squared NCC > 1.
        generator = torch.Generator().manual_seed(9178)
        pred = (1.0 + 1e-4 * torch.randn(1, 1, 32, 32, generator=generator)).requires_grad_()
        target = 1.0 + 1e-4 * torch.randn(1, 1, 32, 32, generator=generator)
        loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=9, smooth_dr=smooth_dr, reduction="none")
        result = loss(pred, target)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(result.min().item(), -1.0)
        self.assertLessEqual(result.max().item(), 0.0)
        result.mean().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    @parameterized.expand([(False,), (True,)])
    def test_uninformative_windows(self, swap):
        # Only one input needs to have insufficient variance to mask the window.
        pred = torch.tensor([[[0.0, 1e-4, 2e-4]]], dtype=torch.float64)
        target = torch.tensor([[[0.0, 1.0, 2.0]]], dtype=torch.float64)
        if swap:
            pred, target = target, pred
        pred.requires_grad_()
        loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=1, reduction="none", smooth_nr=1e-8)
        result = loss(pred, target)
        torch.testing.assert_close(result, torch.zeros_like(result))
        result.sum().backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    @parameterized.expand([0.0, -1e-5, float("nan")])
    def test_nonpositive_smooth_dr(self, smooth_dr):
        with self.assertRaisesRegex(ValueError, "smooth_dr must be strictly positive"):
            LocalNormalizedCrossCorrelationLoss(smooth_dr=smooth_dr)

    def test_variance_at_threshold(self):
        pred = torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True)
        loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=1, smooth_dr=2.0, reduction="none")
        # The center window has variance exactly equal to the threshold.
        result = loss(pred, pred)
        self.assertEqual(result[0, 0, 1].item(), 0.0)
        result[0, 0, 1].backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = LocalNormalizedCrossCorrelationLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=3)
        # spatial_dims unmatch
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(
                torch.ones((1, 3, 3, 3), dtype=torch.float, device=device),
                torch.ones((1, 3, 3, 3), dtype=torch.float, device=device),
            )
        # pred, target shape unmatch
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(
                torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device),
                torch.ones((1, 3, 4, 4, 4), dtype=torch.float, device=device),
            )

    def test_gaussian_kernel_shape_and_symmetry(self):
        # gh-8780: kernel must have correct length, be symmetric, and peak at center
        for kernel_size in [3, 5, 7, 9, 11, 15]:
            k = make_gaussian_kernel(kernel_size)
            self.assertEqual(len(k), kernel_size)
            self.assertTrue(torch.allclose(k, k.flip(0)), f"kernel_size={kernel_size} not symmetric")
            self.assertEqual(k.argmax().item(), kernel_size // 2)
            np.testing.assert_allclose(k.max().item(), 1.0, rtol=1e-6)

    def test_ill_opts(self):
        pred = torch.ones((1, 3, 3, 3, 3), dtype=torch.float)
        target = torch.ones((1, 3, 3, 3, 3), dtype=torch.float)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(kernel_type="unknown")(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(kernel_type=None)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(kernel_size=4)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(reduction="unknown")(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            LocalNormalizedCrossCorrelationLoss(reduction=None)(pred, target)


#     def test_script(self):
#         input_param, input_data, _ = TEST_CASES[0]
#         loss = LocalNormalizedCrossCorrelationLoss(**input_param)
#         test_script_save(loss, input_data["pred"], input_data["target"])

if __name__ == "__main__":
    unittest.main()
