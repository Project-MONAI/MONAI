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

from monai.losses import SoftclDiceLoss, SoftDiceclDiceLoss
from tests.test_utils import skip_if_no_cuda

# Reusable test tensors
ONES_2D = {"input": torch.ones((2, 3, 8, 8)), "target": torch.ones((2, 3, 8, 8))}
ONES_3D = {"input": torch.ones((2, 3, 8, 8, 8)), "target": torch.ones((2, 3, 8, 8, 8))}

# Partial overlap: two 2x2 squares shifted by 1 pixel
PARTIAL_OVERLAP = {
    "input": torch.tensor(
        [[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    ),
    "target": torch.tensor(
        [[[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    ),
}

# Test cases: [loss_params, input_data, expected_value]
CLDICE_CASES = [
    [{}, ONES_2D, 0.0],
    [{}, ONES_3D, 0.0],
    [
        {"sigmoid": True, "smooth_nr": 1e-5, "smooth_dr": 1e-5},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]),
        },
        0.192777,
    ],
    [
        {"softmax": True, "smooth_nr": 1e-5, "smooth_dr": 1e-5},
        {
            "input": torch.tensor([[[[2.0, 0.0], [0.0, 2.0]], [[-2.0, 0.0], [0.0, -2.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]),
        },
        0.148760,
    ],
    [
        {"to_onehot_y": True, "smooth_nr": 1e-5, "smooth_dr": 1e-5},
        {
            "input": torch.tensor([[[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]]),
            "target": torch.tensor([[[[0, 1], [1, 0]]]]),
        },
        0.052631,
    ],
]

COMBINED_CASES = [
    [{"alpha": 0.5}, ONES_2D, 0.0],
    [{"alpha": 0.5, "smooth_nr": 1e-5, "smooth_dr": 1e-5}, PARTIAL_OVERLAP, 0.624995],
    [{"alpha": 0.0, "smooth_nr": 1e-5, "smooth_dr": 1e-5}, PARTIAL_OVERLAP, 0.250000],  # pure Dice
    [{"alpha": 1.0, "smooth_nr": 1e-5, "smooth_dr": 1e-5}, PARTIAL_OVERLAP, 0.999990],  # pure clDice
]

# Test cases for use_checkpoint: [input shape, iter_]
CHECKPOINT_CASES = [[(2, 3, 16, 16), 3], [(2, 2, 12, 12, 12), 5]]


def _loss_and_grad(loss_fn, logits, target):
    """
    Compute a loss and its gradient with respect to the logits.

    Args:
        loss_fn: the loss to call as ``loss_fn(input, target)``.
        logits: the prediction; a copy of it is made to require a gradient.
        target: the ground truth.

    Returns:
        the detached loss and its gradient with respect to ``logits``.
    """
    x = logits.clone().requires_grad_(True)
    loss = loss_fn(x, target)
    (grad,) = torch.autograd.grad(loss, x)
    return loss.detach(), grad


class TestSoftclDiceLoss(unittest.TestCase):
    @parameterized.expand(CLDICE_CASES)
    def test_result(self, loss_params, input_data, expected_val):
        loss = SoftclDiceLoss(**loss_params)
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    @parameterized.expand(CHECKPOINT_CASES)
    def test_use_checkpoint_same_loss_and_grad(self, shape, iter_):
        """``use_checkpoint=True`` gives the same loss and gradient, also under ``torch.no_grad()``."""
        torch.manual_seed(0)
        logits = torch.randn(shape)
        target = (torch.rand(shape) > 0.5).float()
        expected = _loss_and_grad(SoftclDiceLoss(iter_=iter_, sigmoid=True), logits, target)
        loss = SoftclDiceLoss(iter_=iter_, sigmoid=True, use_checkpoint=True)
        for result, expected_val in zip(_loss_and_grad(loss, logits, target), expected):
            np.testing.assert_allclose(result.numpy(), expected_val.numpy(), rtol=1e-5, atol=1e-7)
        with torch.no_grad():  # no gradient needed: falls back to the plain computation
            np.testing.assert_allclose(loss(logits, target).numpy(), expected[0].numpy(), rtol=1e-5, atol=1e-7)

    def test_use_checkpoint_saves_memory(self):
        """``use_checkpoint=True`` keeps far fewer bytes for the backward pass."""
        logits = torch.randn((1, 1, 16, 16, 16), requires_grad=True)
        target = (torch.rand((1, 1, 16, 16, 16)) > 0.5).float()

        def saved_bytes(use_checkpoint):
            """
            Count the bytes that autograd keeps for the backward pass of one forward call.

            This is the total size of the distinct storages saved for backward, not the peak memory.

            Args:
                use_checkpoint: passed to ``SoftclDiceLoss``.

            Returns:
                the number of bytes.
            """
            storages = {}

            def pack(t):
                """
                Record the storage of a tensor that autograd saves for the backward pass.

                Args:
                    t: the saved tensor.

                Returns:
                    ``t`` unchanged.
                """
                storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
                return t

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                SoftclDiceLoss(iter_=5, sigmoid=True, use_checkpoint=use_checkpoint)(logits, target)
            return sum(storages.values())

        self.assertLess(saved_bytes(True), saved_bytes(False) / 4)

    @skip_if_no_cuda
    def test_cuda(self):
        loss = SoftclDiceLoss()
        result = loss(ONES_2D["input"].cuda(), ONES_2D["target"].cuda())
        np.testing.assert_allclose(result.detach().cpu().numpy(), 0.0, atol=1e-4)

    def test_reduction_shapes(self):
        input_tensor = torch.ones((4, 2, 8, 8))
        target = torch.ones((4, 2, 8, 8))

        self.assertEqual(SoftclDiceLoss(reduction="mean")(input_tensor, target).shape, torch.Size([]))
        self.assertEqual(SoftclDiceLoss(reduction="sum")(input_tensor, target).shape, torch.Size([]))
        self.assertEqual(SoftclDiceLoss(reduction="none")(input_tensor, target).shape, torch.Size([4]))

    def test_ill_shape(self):
        loss = SoftclDiceLoss()
        with self.assertRaisesRegex(AssertionError, "ground truth has different shape"):
            loss(torch.ones((1, 3, 8, 8)), torch.ones((1, 4, 8, 8)))

    def test_invalid_activation_combination(self):
        with self.assertRaises(ValueError):
            SoftclDiceLoss(sigmoid=True, softmax=True)

    def test_invalid_other_act(self):
        with self.assertRaises(TypeError):
            SoftclDiceLoss(other_act="invalid")

    def test_invalid_iter_type(self):
        with self.assertRaises(TypeError):
            SoftclDiceLoss(iter_=3.0)

    def test_invalid_iter_value(self):
        with self.assertRaises(ValueError):
            SoftclDiceLoss(iter_=-1)

    def test_zero_input_is_finite(self):
        loss = SoftclDiceLoss(smooth=1e-7, smooth_dr=1e-5)
        result = loss(torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4)))
        self.assertTrue(torch.isfinite(result).all())

    def test_non_default_smooth_dr_changes_result(self):
        input_tensor = torch.zeros((1, 2, 4, 4))
        target = torch.zeros((1, 2, 4, 4))
        loss_a = SoftclDiceLoss(smooth=1e-7, smooth_dr=1e-3)
        loss_b = SoftclDiceLoss(smooth=1e-7, smooth_dr=1e-5)
        result_a = loss_a(input_tensor, target)
        result_b = loss_b(input_tensor, target)
        self.assertTrue(torch.isfinite(result_a).all())
        self.assertTrue(torch.isfinite(result_b).all())
        self.assertNotAlmostEqual(result_a.item(), result_b.item(), places=5)

    def test_non_overlapping_input_is_finite(self):
        loss = SoftclDiceLoss(smooth=1e-7, smooth_dr=1e-5)
        input_tensor = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
        target = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]])
        result = loss(input_tensor, target)
        self.assertTrue(torch.isfinite(result).all())


class TestSoftDiceclDiceLoss(unittest.TestCase):
    @parameterized.expand(COMBINED_CASES)
    def test_result(self, loss_params, input_data, expected_val):
        loss = SoftDiceclDiceLoss(**loss_params)
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_use_checkpoint_same_loss_and_grad(self):
        """``use_checkpoint=True`` gives the same combined loss and gradient."""
        torch.manual_seed(0)
        logits = torch.randn((2, 2, 12, 12, 12))
        target = (torch.rand((2, 2, 12, 12, 12)) > 0.5).float()
        expected = _loss_and_grad(SoftDiceclDiceLoss(sigmoid=True), logits, target)
        result = _loss_and_grad(SoftDiceclDiceLoss(sigmoid=True, use_checkpoint=True), logits, target)
        for result_val, expected_val in zip(result, expected):
            np.testing.assert_allclose(result_val.numpy(), expected_val.numpy(), rtol=1e-5, atol=1e-7)

    @skip_if_no_cuda
    def test_cuda(self):
        loss = SoftDiceclDiceLoss()
        result = loss(ONES_2D["input"].cuda(), ONES_2D["target"].cuda())
        np.testing.assert_allclose(result.detach().cpu().numpy(), 0.0, atol=1e-4)

    def test_dimension_mismatch(self):
        loss = SoftDiceclDiceLoss()
        with self.assertRaises(ValueError):
            loss(torch.ones(2, 3, 8, 8), torch.ones(2, 3, 8))

    def test_channel_mismatch(self):
        loss = SoftDiceclDiceLoss()
        with self.assertRaises(ValueError):
            loss(torch.ones(2, 3, 8, 8), torch.ones(2, 2, 8, 8))

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            SoftDiceclDiceLoss(alpha=1.5)

    def test_invalid_alpha_negative(self):
        with self.assertRaises(ValueError):
            SoftDiceclDiceLoss(alpha=-0.5)

    def test_zero_input_is_finite(self):
        loss = SoftDiceclDiceLoss(smooth=1e-7, smooth_dr=1e-5)
        result = loss(torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4)))
        self.assertTrue(torch.isfinite(result).all())

    def test_non_default_smooth_dr_changes_result(self):
        input_tensor = torch.zeros((1, 2, 4, 4))
        target = torch.zeros((1, 2, 4, 4))
        loss_a = SoftDiceclDiceLoss(smooth=1e-7, smooth_dr=1e-3)
        loss_b = SoftDiceclDiceLoss(smooth=1e-7, smooth_dr=1e-5)
        result_a = loss_a(input_tensor, target)
        result_b = loss_b(input_tensor, target)
        self.assertTrue(torch.isfinite(result_a).all())
        self.assertTrue(torch.isfinite(result_b).all())
        self.assertNotAlmostEqual(result_a.item(), result_b.item(), places=5)

    def test_non_overlapping_input_is_finite(self):
        loss = SoftDiceclDiceLoss(smooth=1e-7, smooth_dr=1e-5)
        input_tensor = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
        target = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]])
        result = loss(input_tensor, target)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
