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
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.navit import NaViT
from tests.test_utils import skip_if_quick

# Shared default kwargs to reduce duplication across test cases.
DEFAULT_2D_KWARGS = {
    "image_size": 64,
    "patch_size": 16,
    "num_classes": 10,
    "hidden_size": 128,
    "mlp_dim": 256,
    "num_layers": 2,
    "num_heads": 4,
    "in_channels": 3,
    "spatial_dims": 2,
}

DEFAULT_3D_KWARGS = {
    "image_size": 64,
    "patch_size": 16,
    "num_classes": 2,
    "hidden_size": 256,
    "mlp_dim": 512,
    "num_layers": 2,
    "num_heads": 8,
    "in_channels": 1,
    "spatial_dims": 3,
}

# Each entry: (init_kwargs, batched_images_spec, expected_output_shape)
# batched_images_spec is a list of groups; each group is a list of image shape tuples.
TEST_CASES_SHAPE = [
    # 2D single image
    (DEFAULT_2D_KWARGS, [[(3, 64, 64)]], (1, 10)),
    # 2D multiple images in one group
    ({**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1}, [[(1, 64, 64), (1, 32, 32), (1, 64, 32)]], (3, 5)),
    # 2D multiple groups
    (
        {**DEFAULT_2D_KWARGS, "image_size": 96, "num_classes": 8, "hidden_size": 192, "mlp_dim": 384, "num_heads": 6},
        [[(3, 96, 96), (3, 64, 64)], [(3, 80, 80)]],
        (3, 8),
    ),
    # 3D single volume
    (DEFAULT_3D_KWARGS, [[(1, 64, 64, 64)]], (1, 2)),
    # 3D multiple volumes, multiple groups
    (
        {**DEFAULT_3D_KWARGS, "image_size": 96, "num_classes": 3},
        [[(1, 96, 96, 96), (1, 64, 64, 64)], [(1, 80, 96, 80)]],
        (3, 3),
    ),
    # token dropout (float)
    ({**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1, "token_dropout_prob": 0.2}, [[(1, 64, 64)]], (1, 5)),
    # custom dim_head
    ({**DEFAULT_2D_KWARGS, "num_classes": 4, "in_channels": 1, "dim_head": 64}, [[(1, 64, 64)]], (1, 4)),
    # qkv_bias enabled
    ({**DEFAULT_2D_KWARGS, "num_classes": 3, "in_channels": 1, "qkv_bias": True}, [[(1, 64, 64)]], (1, 3)),
    # anisotropic image_size 2D
    ({**DEFAULT_2D_KWARGS, "image_size": (64, 128), "num_classes": 4, "in_channels": 1}, [[(1, 64, 128)]], (1, 4)),
    # anisotropic image_size 3D
    (
        {**DEFAULT_3D_KWARGS, "image_size": (64, 64, 96), "hidden_size": 192, "mlp_dim": 384, "num_heads": 6},
        [[(1, 64, 64, 96)]],
        (1, 2),
    ),
]

# Invalid constructor arguments that should raise ValueError
TEST_CASES_ILL_ARG = [
    # spatial_dims not 2 or 3
    {**DEFAULT_2D_KWARGS, "in_channels": 1, "spatial_dims": 4},
    # hidden_size not divisible by num_heads
    {**DEFAULT_2D_KWARGS, "in_channels": 1, "hidden_size": 100, "num_heads": 7},
    # dropout_rate out of [0, 1]
    {**DEFAULT_2D_KWARGS, "in_channels": 1, "dropout_rate": 1.5},
    # emb_dropout_rate out of [0, 1]
    {**DEFAULT_2D_KWARGS, "in_channels": 1, "emb_dropout_rate": -0.1},
    # token_dropout_prob out of (0, 1) as float
    {**DEFAULT_2D_KWARGS, "in_channels": 1, "token_dropout_prob": 1.5},
    # num_heads zero
    {**DEFAULT_2D_KWARGS, "in_channels": 1, "num_heads": 0},
    # image_size not divisible by patch_size
    {**DEFAULT_2D_KWARGS, "in_channels": 1, "image_size": 50},
]

# Forward-validation cases: (description, image_tensor_shape)
# All use the same base net with in_channels=1, spatial_dims=2
TEST_CASES_FORWARD_VALIDATION = [
    # wrong number of input channels (3 instead of 1)
    ("wrong_channels", (3, 64, 64)),
    # wrong number of spatial dimensions (3D image for 2D net)
    ("wrong_spatial_dims", (1, 64, 64, 64)),
    # spatial size not divisible by patch_size
    ("patch_size_not_divisible", (1, 50, 64)),
]


@skip_if_quick
class TestNaViT(unittest.TestCase):

    @parameterized.expand(TEST_CASES_SHAPE)
    def test_shape(self, input_param, batched_images_spec, expected_shape):
        """Test output shape for various configurations."""
        net = NaViT(**input_param)
        with eval_mode(net):
            batched_images = [[torch.randn(*img_shape) for img_shape in group] for group in batched_images_spec]
            result = net(batched_images)
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([(kwargs,) for kwargs in TEST_CASES_ILL_ARG])
    def test_ill_arg(self, input_param):
        """Test that invalid constructor arguments raise ValueError."""
        with self.assertRaises(ValueError):
            NaViT(**input_param)

    @parameterized.expand(TEST_CASES_FORWARD_VALIDATION)
    def test_forward_validation(self, _, image_shape):
        """Forward pass should raise ValueError for invalid input tensors."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "in_channels": 1, "num_classes": 2})
        net.eval()
        with self.assertRaises(ValueError):
            net([[torch.randn(*image_shape)]])

    def test_auto_grouping(self):
        """Auto-packing with group_images=True should produce correct total output size."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1})
        net.eval()
        flat_images = [torch.randn(1, 64, 64) for _ in range(4)]
        result = net(flat_images, group_images=True, group_max_seq_len=32)
        self.assertEqual(result.shape, (4, 5))

    def test_token_dropout_callable_invoked_during_training(self):
        """Token dropout callable is invoked during training and produces correct shape."""
        call_log: list[tuple] = []

        def recording_dropout(h, w):
            call_log.append((h, w))
            return 0.25

        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1}, token_dropout_prob=recording_dropout)
        net.train()
        result = net([[torch.randn(1, 64, 64)]])
        self.assertEqual(result.shape, (1, 5))
        self.assertGreater(len(call_log), 0, "Token dropout callable was not invoked during training.")

    def test_token_dropout_callable_not_invoked_during_eval(self):
        """Token dropout callable is NOT invoked during eval."""
        call_log: list[tuple] = []

        def recording_dropout(h, w):
            call_log.append((h, w))
            return 0.25

        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1}, token_dropout_prob=recording_dropout)
        net.eval()
        net([[torch.randn(1, 64, 64)]])
        self.assertEqual(len(call_log), 0, "Token dropout callable was invoked during eval mode.")

    def test_token_dropout_produces_different_outputs_in_training(self):
        """With token dropout, different RNG seeds produce different training outputs."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1}, token_dropout_prob=0.5)
        net.train()
        input_data = [[torch.randn(1, 64, 64)]]
        torch.manual_seed(0)
        out1 = net(input_data)
        torch.manual_seed(42)
        out2 = net(input_data)
        self.assertFalse(
            torch.allclose(out1, out2),
            "Token dropout should produce different outputs with different RNG seeds during training.",
        )

    def test_token_dropout_disabled_in_eval(self):
        """Token dropout should not be applied during eval, producing deterministic output."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1}, token_dropout_prob=0.5)
        net.eval()
        input_data = [[torch.randn(1, 64, 64)]]
        out1 = net(input_data)
        out2 = net(input_data)
        self.assertTrue(torch.allclose(out1, out2))

    def test_eval_mode_deterministic(self):
        """In eval mode, outputs should be identical across calls."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 5, "in_channels": 1})
        net.eval()
        input_data = [[torch.randn(1, 64, 64)]]
        out1 = net(input_data)
        out2 = net(input_data)
        self.assertTrue(torch.allclose(out1, out2))

    def test_gradient_flow(self):
        """All trainable parameters should receive gradients after a backward pass."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 3, "in_channels": 1})
        net.train()
        output = net([[torch.randn(1, 64, 64)]])
        output.sum().backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")

    def test_all_parameters_trainable(self):
        """All parameters should be trainable by default."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 3, "in_channels": 1})
        frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
        self.assertEqual(frozen, [], f"Found frozen parameters: {frozen}")

    def test_variable_resolution_beyond_reference(self):
        """Images larger than reference image_size should work via positional encoding clamping."""
        net = NaViT(**{**DEFAULT_2D_KWARGS, "num_classes": 2, "in_channels": 1})
        net.eval()
        result = net([[torch.randn(1, 96, 96)]])
        self.assertEqual(result.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
