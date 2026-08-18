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
from unittest.case import skipUnless

import torch
from parameterized import parameterized

from monai.losses import BoundaryLoss
from monai.utils import optional_import

_, has_scipy = optional_import("scipy")

# Reusable test tensors
ONES_2D = {"input": torch.ones((2, 2, 8, 8)), "target": torch.ones((2, 2, 8, 8))}
ONES_3D = {"input": torch.ones((2, 2, 8, 8, 8)), "target": torch.ones((2, 2, 8, 8, 8))}

# Perfect match: target is a 2x2 square, input matches exactly
PERFECT_MATCH = {
    "input": torch.tensor(
        [[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    ),
    "target": torch.tensor(
        [[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    ),
}

# Partial overlap: two 2x2 squares shifted by 1 pixel
PARTIAL_OVERLAP = {
    "input": torch.tensor(
        [[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    ),
    "target": torch.tensor(
        [[[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    ),
}

# Empty foreground class: target has no foreground in class 1
EMPTY_FOREGROUND = {
    "input": torch.tensor(
        [[[[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]]
    ),
    "target": torch.tensor(
        [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    ),
}

TEST_CASES = []
for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
    # Basic 2D test with sigmoid
    TEST_CASES.append(
        [
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device),
            },
            None,  # Just check it runs, value depends on distance map
        ]
    )
    # Basic 3D test with sigmoid
    TEST_CASES.append(
        [
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[[2.0, -2.0], [-2.0, 2.0]], [[2.0, -2.0], [-2.0, 2.0]]]]], device=device),
                "target": torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]], device=device),
            },
            None,
        ]
    )
    # Multi-class 2D with softmax
    TEST_CASES.append(
        [
            {"include_background": True, "softmax": True},
            {
                "input": torch.tensor([[[[2.0, 0.0], [0.0, 2.0]], [[-2.0, 0.0], [0.0, -2.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]], device=device),
            },
            None,
        ]
    )
    # With to_onehot_y
    TEST_CASES.append(
        [
            {"include_background": True, "to_onehot_y": True, "softmax": True},
            {
                "input": torch.tensor([[[[2.0, 0.0], [0.0, 2.0]], [[-2.0, 0.0], [0.0, -2.0]]]], device=device),
                "target": torch.tensor([[[[0, 0], [0, 1]]]], device=device),
            },
            None,
        ]
    )
    # With reduction="none"
    TEST_CASES.append(
        [
            {"include_background": True, "sigmoid": True, "reduction": "none"},
            {
                "input": torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device),
            },
            None,
        ]
    )
    # With reduction="sum"
    TEST_CASES.append(
        [
            {"include_background": True, "sigmoid": True, "reduction": "sum"},
            {
                "input": torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device),
            },
            None,
        ]
    )
    # Exclude background
    TEST_CASES.append(
        [
            {"include_background": False, "sigmoid": True},
            {
                "input": torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]], [[-2.0, 2.0], [2.0, -2.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]], device=device),
            },
            None,
        ]
    )
    # With other_act
    TEST_CASES.append(
        [
            {"include_background": True, "other_act": torch.tanh},
            {
                "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device),
            },
            None,
        ]
    )


def _describe_test_case(test_func, test_number, params):
    input_param, input_data, _ = params.args
    return f"params:{input_param}, shape:{input_data['input'].shape}, device:{input_data['input'].device}"


@skipUnless(has_scipy, "Scipy required")
class TestBoundaryLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES, doc_func=_describe_test_case)
    def test_runs(self, input_param, input_data, _):
        """Test that the loss runs without errors for various configurations."""
        loss = BoundaryLoss(**input_param)
        result = loss(**input_data)
        # Just verify it's a scalar tensor and finite
        self.assertTrue(torch.isfinite(result).all())

    def test_perfect_match(self):
        """Test that perfect predictions yield lower loss than imperfect ones."""
        loss_fn = BoundaryLoss()
        perfect_loss = loss_fn(PERFECT_MATCH["input"], PERFECT_MATCH["target"])
        partial_loss = loss_fn(PARTIAL_OVERLAP["input"], PARTIAL_OVERLAP["target"])
        # Perfect match should have lower loss than partial overlap
        self.assertLess(perfect_loss.item(), partial_loss.item())

    def test_reduction_shapes(self):
        """Test that different reductions produce expected shapes."""
        input_tensor = torch.ones((4, 2, 8, 8))
        target = torch.ones((4, 2, 8, 8))

        self.assertEqual(BoundaryLoss(reduction="mean")(input_tensor, target).shape, torch.Size([]))
        self.assertEqual(BoundaryLoss(reduction="sum")(input_tensor, target).shape, torch.Size([]))
        # With include_background=True and 2 classes, shape should be (4, 2)
        self.assertEqual(BoundaryLoss(reduction="none")(input_tensor, target).shape, torch.Size([4, 2]))

    def test_reduction_shapes_exclude_background(self):
        """Test shapes when background is excluded."""
        input_tensor = torch.ones((4, 3, 8, 8))
        target = torch.ones((4, 3, 8, 8))

        # With include_background=False, shape should be (4, 2) for 3 classes
        self.assertEqual(
            BoundaryLoss(reduction="none", include_background=False)(input_tensor, target).shape, torch.Size([4, 2])
        )

    def test_single_channel_options_warn_and_are_ignored(self):
        """Test that single-channel-only options follow other MONAI loss behavior."""
        input_tensor = torch.randn((1, 1, 4, 4), requires_grad=True)
        target = torch.zeros((1, 1, 4, 4))
        target[..., 1:3, 1:3] = 1

        with self.assertWarns(Warning):
            loss = BoundaryLoss(softmax=True)(input_tensor, target)
        loss.backward()
        self.assertGreater(input_tensor.grad.abs().sum().item(), 0.0)

        with self.assertWarns(Warning):
            result = BoundaryLoss(include_background=False)(input_tensor.detach(), target)
        self.assertTrue(torch.isfinite(result))

        with self.assertWarns(Warning):
            result = BoundaryLoss(to_onehot_y=True)(input_tensor.detach(), target)
        self.assertTrue(torch.isfinite(result))

    def test_to_onehot_y_accepts_channel_free_target(self):
        """Test target labels can omit the singleton channel dimension."""
        input_tensor = torch.randn((2, 3, 4, 4))
        target = torch.randint(0, 3, size=(2, 4, 4))
        result = BoundaryLoss(to_onehot_y=True, softmax=True)(input_tensor, target)
        self.assertTrue(torch.isfinite(result))

    def test_degenerate_target_distance_map_is_zero(self):
        """Test that empty and full classes don't create edge-biased distance maps."""
        loss_fn = BoundaryLoss()
        empty_target = torch.zeros((1, 1, 4, 4))
        full_target = torch.ones((1, 1, 4, 4))

        self.assertTrue(torch.equal(loss_fn.compute_distance_map(empty_target), torch.zeros_like(empty_target)))
        self.assertTrue(torch.equal(loss_fn.compute_distance_map(full_target), torch.zeros_like(full_target)))

    def test_batch_reduction_changes_none_shape_and_values(self):
        """Test that batch=True reduces the batch dimension before final reduction."""
        input_tensor = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]]])
        target = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]])

        batch_false = BoundaryLoss(reduction="none", batch=False)(input_tensor, target)
        batch_true = BoundaryLoss(reduction="none", batch=True)(input_tensor, target)

        self.assertEqual(batch_false.shape, torch.Size([2, 1]))
        self.assertEqual(batch_true.shape, torch.Size([1]))
        self.assertTrue(torch.allclose(batch_true, batch_false.mean(dim=0)))

    def test_ill_shape(self):
        """Test that mismatched shapes raise an error."""
        loss = BoundaryLoss()
        with self.assertRaisesRegex(AssertionError, "shapes do not match"):
            loss(torch.ones((1, 1, 2, 3)), torch.ones((1, 4, 5, 6)))

    def test_ill_opts(self):
        """Test that invalid options raise errors."""
        with self.assertRaisesRegex(ValueError, ""):
            BoundaryLoss(sigmoid=True, softmax=True)
        with self.assertRaisesRegex(ValueError, ""):
            BoundaryLoss(sigmoid=True, other_act=torch.tanh)
        with self.assertRaisesRegex(ValueError, ""):
            BoundaryLoss(softmax=True, other_act=torch.tanh)
        with self.assertRaisesRegex(ValueError, ""):
            BoundaryLoss(sigmoid=True, softmax=True, other_act=torch.tanh)

        chn_input = torch.ones((1, 1, 3, 3))
        chn_target = torch.ones((1, 1, 3, 3))
        with self.assertRaisesRegex(ValueError, ""):
            BoundaryLoss(reduction="unknown")(chn_input, chn_target)

    def test_invalid_other_act_type(self):
        """Test that non-callable other_act raises TypeError."""
        with self.assertRaises(TypeError):
            BoundaryLoss(other_act="invalid")

    def test_empty_foreground(self):
        """Test that empty foreground classes don't crash the loss."""
        loss_fn = BoundaryLoss(sigmoid=False)
        result = loss_fn(EMPTY_FOREGROUND["input"], EMPTY_FOREGROUND["target"])
        self.assertTrue(torch.isfinite(result))

    def test_dimension_validation(self):
        """Test that unsupported dimensions raise errors."""
        loss = BoundaryLoss()
        with self.assertRaises(ValueError):
            # 1D input should fail
            loss(torch.ones((1, 1, 10)), torch.ones((1, 1, 10)))
        with self.assertRaises(ValueError):
            # 4D input (5D with batch+channel) should fail
            loss(torch.ones((1, 1, 2, 2, 2, 2)), torch.ones((1, 1, 2, 2, 2, 2)))

    def test_distance_map_computation(self):
        """Test that distance maps are computed correctly for a simple case."""
        # Simple 3x3 case: foreground in center pixel
        target = torch.zeros((1, 1, 3, 3))
        target[0, 0, 1, 1] = 1.0  # Center pixel is foreground

        loss_fn = BoundaryLoss()
        distance_map = loss_fn.compute_distance_map(target)

        # Center pixel is on the boundary (single-pixel object), so distance should be 0 or near 0
        self.assertAlmostEqual(distance_map[0, 0, 1, 1].item(), 0.0, places=5)

        # Corners should be positive (outside foreground)
        self.assertGreater(distance_map[0, 0, 0, 0].item(), 0)
        self.assertGreater(distance_map[0, 0, 0, 2].item(), 0)
        self.assertGreater(distance_map[0, 0, 2, 0].item(), 0)
        self.assertGreater(distance_map[0, 0, 2, 2].item(), 0)

        # Neighbors of center should also be positive (outside foreground)
        self.assertGreater(distance_map[0, 0, 0, 1].item(), 0)
        self.assertGreater(distance_map[0, 0, 1, 0].item(), 0)

    def test_loss_gradient_flow(self):
        """Test that gradients flow through the loss."""
        input_tensor = torch.randn((2, 2, 8, 8), requires_grad=True)
        target = torch.ones((2, 2, 8, 8))

        loss_fn = BoundaryLoss(sigmoid=True)
        loss = loss_fn(input_tensor, target)
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.isfinite(input_tensor.grad).all())

    def test_consistency_with_hausdorff_loss(self):
        """Test that BoundaryLoss behaves differently from HausdorffDTLoss on the same input."""
        from monai.losses import HausdorffDTLoss

        input_tensor = torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

        bl_loss = BoundaryLoss(sigmoid=True)(input_tensor, target)
        hd_loss = HausdorffDTLoss(sigmoid=True)(input_tensor, target)

        # They should produce different values (different formulations)
        self.assertNotAlmostEqual(bl_loss.item(), hd_loss.item(), places=3)


if __name__ == "__main__":
    unittest.main()
