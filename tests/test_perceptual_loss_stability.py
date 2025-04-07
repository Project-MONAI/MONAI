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
import torch.optim as optim
from parameterized import parameterized

from monai.losses.perceptual import normalize_tensor
from monai.utils import set_determinism


class TestNormalizeTensorStability(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=0)
        self.addCleanup(set_determinism, None)

    def tearDown(self):
        set_determinism(None)

    @parameterized.expand([["e-3", 1e-3], ["e-6", 1e-6], ["e-9", 1e-9], ["e-12", 1e-12]])  # Small values
    def test_normalize_tensor_stability(self, name, scale):
        """Test that small values don't produce NaNs + are handled gracefully."""
        # Create tensor
        x = torch.zeros(2, 3, 10, 10, requires_grad=True)

        optimizer = optim.Adam([x], lr=0.01)
        x_scaled = x * scale
        normalized = normalize_tensor(x_scaled)

        # Compute to force backward pass
        loss = normalized.sum()

        # this is where it failed before
        loss.backward()

        # Check for NaNs in gradients
        self.assertFalse(torch.isnan(x.grad).any(), f"NaN gradients detected with scale {scale:.10e}")

    def test_normalize_tensor_zero_input(self):
        """Test that normalize_tensor handles zero inputs gracefully."""
        # Create tensor with zeros
        x = torch.zeros(2, 3, 4, 4, requires_grad=True)

        normalized = normalize_tensor(x)
        loss = normalized.sum()
        loss.backward()

        # Check for NaNs in gradients
        self.assertFalse(torch.isnan(x.grad).any(), "NaN gradients detected with zero input")


if __name__ == "__main__":
    unittest.main()
