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

from monai.losses import AUCMLoss
from tests.test_utils import test_script_save

TEST_CASES = [
    [{"version": "v1"}, {"input": torch.tensor([[1.0], [2.0]]), "target": torch.tensor([[1.0], [0.0]])}, 2.375000],
    [{"version": "v2"}, {"input": torch.tensor([[1.0], [2.0]]), "target": torch.tensor([[1.0], [0.0]])}, 9.500000],
    # ------------------------------------------------------------------
    # Explicit imratio coverage for v1
    # ------------------------------------------------------------------
    [
        {"version": "v1", "imratio": 0.25},
        {"input": torch.tensor([[0.0], [1.0], [2.0], [3.0]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        1.687500,
    ],
    [
        {"version": "v1", "imratio": 0.5},
        {"input": torch.tensor([[0.0], [1.0], [2.0], [3.0]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        3.625000,
    ],
    [
        {"version": "v1", "imratio": 0.75},
        {"input": torch.tensor([[0.0], [1.0], [2.0], [3.0]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        5.437500,
    ],
    # ------------------------------------------------------------------
    # imratio ignored in v2
    # ------------------------------------------------------------------
    [
        {"version": "v2", "imratio": 0.25},
        {"input": torch.tensor([[0.0], [1.0], [2.0], [3.0]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        14.500000,
    ],
    [
        {"version": "v2", "imratio": 0.75},
        {"input": torch.tensor([[0.0], [1.0], [2.0], [3.0]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        14.500000,
    ],
    # ------------------------------------------------------------------
    # Margin coverage for v1
    # ------------------------------------------------------------------
    [
        {"version": "v1", "margin": 0.5},
        {"input": torch.tensor([[2.0], [0.5], [-1.0], [-0.5]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        -0.687500,
    ],
    [
        {"version": "v1", "margin": 2.0},
        {"input": torch.tensor([[2.0], [0.5], [-1.0], [-0.5]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        0.062500,
    ],
    # ------------------------------------------------------------------
    # Combined imratio + margin coverage
    # ------------------------------------------------------------------
    [
        {"version": "v1", "imratio": 0.25, "margin": 0.5},
        {"input": torch.tensor([[2.0], [0.5], [-1.0], [-0.5]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        -0.687500,
    ],
    [
        {"version": "v2", "imratio": 0.25, "margin": 0.5},
        {"input": torch.tensor([[2.0], [0.5], [-1.0], [-0.5]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        -2.750000,
    ],
    # ------------------------------------------------------------------
    # Margin coverage for v2
    # ------------------------------------------------------------------
    [
        {"version": "v2", "margin": 0.5},
        {"input": torch.tensor([[2.0], [0.5], [-1.0], [-0.5]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        -2.750000,
    ],
    [
        {"version": "v2", "margin": 2.0},
        {"input": torch.tensor([[2.0], [0.5], [-1.0], [-0.5]]), "target": torch.tensor([[1.0], [1.0], [0.0], [0.0]])},
        0.250000,
    ],
    # ------------------------------------------------------------------
    # Blank / degenerate inputs
    # ------------------------------------------------------------------
    [{"version": "v1"}, {"input": torch.zeros((4, 1)), "target": torch.tensor([[1.0], [0.0], [1.0], [0.0]])}, 0.375000],
    # ------------------------------------------------------------------
    # Higher-dimensional tensors
    # ------------------------------------------------------------------
    [
        {"version": "v1"},
        {"input": torch.tensor([[[[2.0, -1.0], [0.5, -0.5]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])},
        -0.437500,
    ],
    [
        {"version": "v2"},
        {"input": torch.tensor([[[[2.0, -1.0], [0.5, -0.5]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])},
        -1.750000,
    ],
]

BAD_ARGS = [[{"version": "invalid"}], [{"imratio": -0.1}], [{"imratio": 1.5}], [{"reduction": "invalid"}]]


SHAPE_ERROR_CASES = [
    [torch.randn(32), torch.randint(0, 2, (32, 1)).float()],
    [torch.randn(32, 2), torch.randint(0, 2, (32, 1)).float()],
    [torch.randn(32, 1), torch.randint(0, 2, (32, 2)).float()],
    [torch.randn(32, 1), torch.randint(0, 2, (16, 1)).float()],
]


class TestAUCMLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_forward_values(self, input_param, input_data, expected_val):
        loss_fn = AUCMLoss(**input_param)

        # ------------------------------------------------------------
        # Set deterministic non-zero internal optimization variables
        # to make margin-dependent behavior testable
        # ------------------------------------------------------------
        loss_fn.a.data.fill_(0.5)
        loss_fn.b.data.fill_(-0.5)
        loss_fn.alpha.data.fill_(1.0)

        result = loss_fn.forward(**input_data)

        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5, atol=1e-5)

    @parameterized.expand(BAD_ARGS)
    def test_bad_args(self, kwargs):
        with self.assertRaises((ValueError, TypeError)):
            AUCMLoss(**kwargs)

    @parameterized.expand(SHAPE_ERROR_CASES)
    def test_invalid_shapes(self, pred, target):
        with self.assertRaises(ValueError):
            AUCMLoss()(pred, target)

    @parameterized.expand([("v1",), ("v2",)])
    def test_all_negative_batch(self, version):
        pred = torch.zeros((8, 1))
        target = torch.zeros((8, 1))

        loss = AUCMLoss(version=version)(pred, target)

        self.assertTrue(torch.isfinite(loss))

    def test_non_binary_target(self):
        pred = torch.randn(32, 1)

        target = torch.tensor([[0.5], [1.0], [2.0], [0.0]] * 8)

        with self.assertRaises(ValueError):
            AUCMLoss()(pred, target)

    @parameterized.expand([("v1",), ("v2",)])
    def test_backward(self, version):
        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randint(0, 2, (32, 1)).float()

        loss = AUCMLoss(version=version)(pred, target)

        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    @parameterized.expand([("v1",), ("v2",)])
    def test_blank_predictions_mixed_targets(self, version):
        pred = torch.zeros((4, 1))
        target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        loss = AUCMLoss(version=version)(pred, target)
        if version == "v1":
            self.assertTrue(torch.isfinite(loss))
        else:
            self.assertTrue(torch.isfinite(loss) or torch.isnan(loss))

    @parameterized.expand([("mean",), ("sum",), ("none",)])
    def test_reduction_argument(self, reduction):
        pred = torch.tensor([[1.0], [2.0]])
        target = torch.tensor([[1.0], [0.0]])

        loss = AUCMLoss(reduction=reduction)(pred, target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_script_save(self):
        loss_fn = AUCMLoss()

        test_script_save(loss_fn, torch.randn(32, 1), torch.randint(0, 2, (32, 1)).float())


if __name__ == "__main__":
    unittest.main()
