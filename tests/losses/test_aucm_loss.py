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
    # small deterministic cases (with expected values)
    ("v1", torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [0.0]]), 1.25),
    ("v2", torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [0.0]]), 5.0),
]


class TestAUCMLoss(unittest.TestCase):
    """Unit tests for AUCMLoss covering correctness, edge cases, and scriptability."""

    @parameterized.expand([("v1",), ("v2",)])
    def test_versions(self, version):
        """Test AUCMLoss with different versions."""
        loss_fn = AUCMLoss(version=version)
        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randint(0, 2, (32, 1)).float()
        loss = loss_fn(pred, target)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    @parameterized.expand(TEST_CASES)
    def test_known_values(self, version, pred, target, expected):
        """Test AUCMLoss against fixed manually computed values."""
        loss = AUCMLoss(version=version)(pred, target)
        np.testing.assert_allclose(loss.detach().cpu().numpy(), expected, atol=1e-5, rtol=1e-5)

    @parameterized.expand([("v1",), ("v2",)])
    def test_high_dimensional(self, version):
        """Test AUCMLoss with higher dimensional preds (e.g., segmentation)."""
        loss_fn = AUCMLoss(version=version)

        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 8, 8)).float()

        loss = loss_fn(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_imbalanced(self):
        """Test AUCMLoss with highly imbalanced targets."""
        loss_fn = AUCMLoss(version="v1")

        pred = torch.randn(32, 1)
        target = torch.zeros(32, 1)
        target[0] = 1.0  # only one positive

        loss = loss_fn(pred, target)

        self.assertIsInstance(loss, torch.Tensor)

    def test_invalid_version(self):
        """Test that invalid version raises ValueError."""
        with self.assertRaises(ValueError):
            AUCMLoss(version="invalid")

    def test_invalid_imratio(self):
        """Test that invalid imratio raises ValueError."""
        with self.assertRaises(ValueError):
            AUCMLoss(imratio=1.5)
        with self.assertRaises(ValueError):
            AUCMLoss(imratio=-0.1)

    def test_invalid_pred_shape(self):
        """Test that invalid pred shape raises ValueError."""
        loss_fn = AUCMLoss()
        pred = torch.randn(32, 2)  # Wrong channel
        target = torch.randint(0, 2, (32, 1)).float()
        with self.assertRaises(ValueError):
            loss_fn(pred, target)

    def test_invalid_target_shape(self):
        """Test that invalid target shape raises ValueError."""
        loss_fn = AUCMLoss()
        pred = torch.randn(32, 1)
        target = torch.randint(0, 2, (32, 2)).float()  # Wrong channel
        with self.assertRaises(ValueError):
            loss_fn(pred, target)

    def test_insufficient_dimensions(self):
        """Test that tensors with insufficient dimensions raise ValueError."""
        loss_fn = AUCMLoss()
        pred = torch.randn(32)  # 1D tensor
        target = torch.randint(0, 2, (32, 1)).float()
        with self.assertRaises(ValueError):
            loss_fn(pred, target)

    def test_shape_mismatch(self):
        """Test that mismatched shapes raise ValueError."""
        loss_fn = AUCMLoss()
        pred = torch.randn(32, 1)
        target = torch.randint(0, 2, (16, 1)).float()
        with self.assertRaises(ValueError):
            loss_fn(pred, target)

    def test_non_binary_target(self):
        """Test that non-binary target values raise ValueError."""
        loss_fn = AUCMLoss()
        pred = torch.randn(32, 1)
        target = torch.tensor([[0.5], [1.0], [2.0], [0.0]] * 8)  # 32x1, still non-binary
        with self.assertRaises(ValueError):
            loss_fn(pred, target)

    def test_backward(self):
        """Test that gradients can be computed."""
        loss_fn = AUCMLoss()
        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randint(0, 2, (32, 1)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_script_save(self):
        """Test that the loss can be saved as TorchScript."""
        loss_fn = AUCMLoss()
        test_script_save(loss_fn, torch.randn(32, 1), torch.randint(0, 2, (32, 1)).float())


if __name__ == "__main__":
    unittest.main()
