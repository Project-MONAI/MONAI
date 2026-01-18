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

from monai.losses import AsymmetricUnifiedFocalLoss

TEST_CASES = [
    # Case 0: Binary Classification, Perfect Prediction (Probs)
    # Input is already probabilities (use_softmax=False), perfect prediction -> Loss should be close to 0
    [
        {
            "init_kwargs": {"use_softmax": False, "to_onehot_y": False},
            "forward_kwargs": {
                "input": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]),
                "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]),
            },
        },
        0.0,
    ],
    # Case 1: Multi-class (3 Classes), Perfect Prediction (Logits)
    # Input is Logits (use_softmax=True), large value difference implies high confidence -> Loss should be close to 0
    [
        {
            "init_kwargs": {"use_softmax": True, "to_onehot_y": False},
            "forward_kwargs": {
                # Logits: Large positive values indicate high probability
                "input": torch.tensor(
                    [[[[10.0, -10.0], [-10.0, -10.0]], [[-10.0, 10.0], [-10.0, -10.0]], [[-10.0, -10.0], [10.0, 10.0]]]]
                ),
                "target": torch.tensor(
                    [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]]
                ),
            },
        },
        0.0,
    ],
    # Case 2: Label Indices Input (to_onehot_y=True)
    # Test automatic conversion from Index to One-Hot
    [
        {
            "init_kwargs": {"use_softmax": False, "to_onehot_y": True},
            "forward_kwargs": {
                "input": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]]),
                "target": torch.tensor([[[[0, 1], [2, 2]]]]),  # Shape (1, 1, 2, 2)
            },
        },
        0.0,
    ],
]

TEST_CASES_REDUCTION = [
    # Case: Reduction = 'none'
    # Output shape should be (B, C)
    [
        {
            "init_kwargs": {"reduction": "none", "use_softmax": False},
            "forward_kwargs": {
                "input": torch.randn(2, 3, 4, 4).sigmoid(),  # B=2, C=3
                "target": torch.randint(0, 2, (2, 3, 4, 4)).float(),
            },
        },
        (2, 3),
    ],
    # Case: Reduction = 'none' AND include_background=False
    # Output shape should be (B, C-1) -> (2, 2)
    [
        {
            "init_kwargs": {"reduction": "none", "include_background": False, "use_softmax": False},
            "forward_kwargs": {
                "input": torch.randn(2, 3, 4, 4).sigmoid(),
                "target": torch.randint(0, 2, (2, 3, 4, 4)).float(),
            },
        },
        (2, 2),
    ],
]


class TestAsymmetricUnifiedFocalLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_result(self, input_params, expected_val):
        """Test numerical accuracy of the loss."""
        init_kwargs = input_params.get("init_kwargs", {})
        forward_kwargs = input_params.get("forward_kwargs", {})

        loss_func = AsymmetricUnifiedFocalLoss(**init_kwargs)
        result = loss_func(**forward_kwargs)

        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-3, rtol=1e-4)

    @parameterized.expand(TEST_CASES_REDUCTION)
    def test_reduction_shape(self, input_params, expected_shape):
        """Test output shapes under different Reduction modes."""
        init_kwargs = input_params.get("init_kwargs", {})
        forward_kwargs = input_params.get("forward_kwargs", {})

        loss_func = AsymmetricUnifiedFocalLoss(**init_kwargs)
        result = loss_func(**forward_kwargs)

        self.assertEqual(result.shape, expected_shape, msg=f"Expected shape {expected_shape} but got {result.shape}")

    def test_ill_shape(self):
        """Test handling of incorrect shapes."""
        loss_func = AsymmetricUnifiedFocalLoss()
        with self.assertRaisesRegex(ValueError, "ground truth has different shape"):
            loss_func(torch.ones((2, 2, 2)), torch.ones((2, 2, 2, 2)))

    def test_mismatch_shape(self):
        """Test completely mismatched input and target shapes."""
        loss_func = AsymmetricUnifiedFocalLoss()
        with self.assertRaisesRegex(ValueError, "ground truth has different shape"):
            loss_func(torch.ones((1, 2, 4, 4)), torch.ones((1, 2, 3, 3)))

    def test_script(self):
        """Test TorchScript compatibility."""
        loss_func = AsymmetricUnifiedFocalLoss()
        input_data = torch.rand(1, 2, 4, 4)
        target_data = torch.rand(1, 2, 4, 4)
        try:
            scripted_loss = torch.jit.script(loss_func)
            scripted_loss(input_data, target_data)
        except Exception as e:
            self.fail(f"TorchScript failed: {e}")

    def test_with_cuda(self):
        """Test CUDA support."""
        if not torch.cuda.is_available():
            return

        loss_func = AsymmetricUnifiedFocalLoss()
        input_data = torch.rand(1, 2, 4, 4).cuda()
        target_data = torch.rand(1, 2, 4, 4).cuda()

        try:
            output = loss_func(input_data, target_data)
            self.assertTrue(output.is_cuda, "Output should be on CUDA")
        except Exception as e:
            self.fail(f"CUDA forward pass failed: {e}")


if __name__ == "__main__":
    unittest.main()
