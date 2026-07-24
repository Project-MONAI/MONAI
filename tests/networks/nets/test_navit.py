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

# Each entry: (init_kwargs, batched_images_spec, expected_output_shape)
# batched_images_spec is a list of groups; each group is a list of image shape tuples.
TEST_CASES_SHAPE = [
    # 2D single image
    (
        {
            "image_size": 64,
            "patch_size": 16,
            "num_classes": 10,
            "hidden_size": 128,
            "mlp_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "in_channels": 3,
            "spatial_dims": 2,
        },
        [[(3, 64, 64)]],
        (1, 10),
    ),
    # 2D multiple images in one group
    (
        {
            "image_size": 64,
            "patch_size": 16,
            "num_classes": 5,
            "hidden_size": 128,
            "mlp_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "in_channels": 1,
            "spatial_dims": 2,
        },
        [[(1, 64, 64), (1, 32, 32), (1, 64, 32)]],
        (3, 5),
    ),
    # 2D multiple groups
    (
        {
            "image_size": 96,
            "patch_size": 16,
            "num_classes": 8,
            "hidden_size": 192,
            "mlp_dim": 384,
            "num_layers": 2,
            "num_heads": 6,
            "in_channels": 3,
            "spatial_dims": 2,
        },
        [[(3, 96, 96), (3, 64, 64)], [(3, 80, 80)]],
        (3, 8),
    ),
    # 3D single volume
    (
        {
            "image_size": 64,
            "patch_size": 16,
            "num_classes": 2,
            "hidden_size": 256,
            "mlp_dim": 512,
            "num_layers": 2,
            "num_heads": 8,
            "in_channels": 1,
            "spatial_dims": 3,
        },
        [[(1, 64, 64, 64)]],
        (1, 2),
    ),
    # 3D multiple volumes, multiple groups
    (
        {
            "image_size": 96,
            "patch_size": 16,
            "num_classes": 3,
            "hidden_size": 256,
            "mlp_dim": 512,
            "num_layers": 2,
            "num_heads": 8,
            "in_channels": 1,
            "spatial_dims": 3,
        },
        [[(1, 96, 96, 96), (1, 64, 64, 64)], [(1, 80, 96, 80)]],
        (3, 3),
    ),
    # token dropout (float)
    (
        {
            "image_size": 64,
            "patch_size": 16,
            "num_classes": 5,
            "hidden_size": 128,
            "mlp_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "in_channels": 1,
            "spatial_dims": 2,
            "token_dropout_prob": 0.2,
        },
        [[(1, 64, 64)]],
        (1, 5),
    ),
    # custom dim_head
    (
        {
            "image_size": 64,
            "patch_size": 16,
            "num_classes": 4,
            "hidden_size": 128,
            "mlp_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "in_channels": 1,
            "spatial_dims": 2,
            "dim_head": 64,
        },
        [[(1, 64, 64)]],
        (1, 4),
    ),
    # qkv_bias enabled
    (
        {
            "image_size": 64,
            "patch_size": 16,
            "num_classes": 3,
            "hidden_size": 128,
            "mlp_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "in_channels": 1,
            "spatial_dims": 2,
            "qkv_bias": True,
        },
        [[(1, 64, 64)]],
        (1, 3),
    ),
    # anisotropic image_size 2D
    (
        {
            "image_size": (64, 128),
            "patch_size": 16,
            "num_classes": 4,
            "hidden_size": 128,
            "mlp_dim": 256,
            "num_layers": 2,
            "num_heads": 4,
            "in_channels": 1,
            "spatial_dims": 2,
        },
        [[(1, 64, 128)]],
        (1, 4),
    ),
    # anisotropic image_size 3D
    (
        {
            "image_size": (64, 64, 96),
            "patch_size": 16,
            "num_classes": 2,
            "hidden_size": 192,
            "mlp_dim": 384,
            "num_layers": 2,
            "num_heads": 6,
            "in_channels": 1,
            "spatial_dims": 3,
        },
        [[(1, 64, 64, 96)]],
        (1, 2),
    ),
]

# Invalid constructor arguments that should raise ValueError
TEST_CASES_ILL_ARG = [
    # spatial_dims not 2 or 3
    {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 2,
        "hidden_size": 128,
        "mlp_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "in_channels": 1,
        "spatial_dims": 4,
    },
    # hidden_size not divisible by num_heads
    {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 2,
        "hidden_size": 100,
        "mlp_dim": 256,
        "num_layers": 2,
        "num_heads": 7,
        "in_channels": 1,
        "spatial_dims": 2,
    },
    # dropout_rate out of [0, 1]
    {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 2,
        "hidden_size": 128,
        "mlp_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "in_channels": 1,
        "spatial_dims": 2,
        "dropout_rate": 1.5,
    },
    # emb_dropout_rate out of [0, 1]
    {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 2,
        "hidden_size": 128,
        "mlp_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "in_channels": 1,
        "spatial_dims": 2,
        "emb_dropout_rate": -0.1,
    },
    # token_dropout_prob out of (0, 1) as float
    {
        "image_size": 64,
        "patch_size": 16,
        "num_classes": 2,
        "hidden_size": 128,
        "mlp_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "in_channels": 1,
        "spatial_dims": 2,
        "token_dropout_prob": 1.5,
    },
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

    def test_forward_validation_wrong_channels(self):
        """Forward pass should raise ValueError for wrong number of input channels."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=2,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        net.eval()
        with self.assertRaises(ValueError):
            net([[torch.randn(3, 64, 64)]])

    def test_forward_validation_wrong_spatial_dims(self):
        """Forward pass should raise ValueError when image ndim doesn't match spatial_dims."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=2,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        net.eval()
        with self.assertRaises(ValueError):
            net([[torch.randn(1, 64, 64, 64)]])

    def test_forward_validation_patch_size_not_divisible(self):
        """Forward pass should raise ValueError when spatial size is not divisible by patch_size."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=2,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        net.eval()
        with self.assertRaises(ValueError):
            net([[torch.randn(1, 50, 64)]])

    def test_auto_grouping(self):
        """Auto-packing with group_images=True should produce correct total output size."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=5,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        net.eval()
        flat_images = [torch.randn(1, 64, 64) for _ in range(4)]
        result = net(flat_images, group_images=True, group_max_seq_len=32)
        self.assertEqual(result.shape, (4, 5))

    def test_token_dropout_callable(self):
        """Token dropout with a callable should produce correctly shaped output."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=5,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
            token_dropout_prob=lambda h, w: 0.1 if h > 48 else 0.0,
        )
        net.eval()
        result = net([[torch.randn(1, 64, 64)]])
        self.assertEqual(result.shape, (1, 5))

    def test_eval_mode_deterministic(self):
        """In eval mode with the same seed, outputs should be identical."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=5,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        net.eval()
        input_data = [[torch.randn(1, 64, 64)]]
        torch.manual_seed(0)
        out1 = net(input_data)
        torch.manual_seed(0)
        out2 = net(input_data)
        self.assertTrue(torch.allclose(out1, out2))

    def test_gradient_flow(self):
        """All trainable parameters should receive gradients after a backward pass."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=3,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        net.train()
        output = net([[torch.randn(1, 64, 64)]])
        output.sum().backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")

    def test_all_parameters_trainable(self):
        """All parameters should be trainable by default."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=3,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
        self.assertEqual(frozen, [], f"Found frozen parameters: {frozen}")

    def test_variable_resolution_beyond_reference(self):
        """Images larger than reference image_size should work via positional encoding clamping."""
        net = NaViT(
            image_size=64,
            patch_size=16,
            num_classes=2,
            hidden_size=128,
            mlp_dim=256,
            num_layers=2,
            num_heads=4,
            in_channels=1,
            spatial_dims=2,
        )
        net.eval()
        result = net([[torch.randn(1, 96, 96)]])
        self.assertEqual(result.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
