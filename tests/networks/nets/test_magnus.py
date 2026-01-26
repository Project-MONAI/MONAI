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

"""
Unit tests for MAGNUS network.

To run tests:
    pytest test_magnus.py -v

Or with unittest:
    python -m pytest test_magnus.py -v
"""

from __future__ import annotations

import unittest

import torch
from parameterized import parameterized

from monai.networks.nets.magnus import (
    MAGNUS,
    CNNPath,
    CrossModalAttentionFusion,
    DecoderBlock,
    ScaleAdaptiveConv,
    SEBlock,
    TransformerPath,
)

# Test cases for MAGNUS model
MAGNUS_TEST_CASES = [
    # (spatial_dims, in_channels, out_channels, input_shape, expected_output_shape)
    (3, 1, 2, (1, 1, 64, 64, 64), (1, 2, 64, 64, 64)),
    (3, 4, 3, (2, 4, 32, 32, 32), (2, 3, 32, 32, 32)),
    (2, 1, 2, (1, 1, 128, 128), (1, 2, 128, 128)),
    (2, 3, 5, (2, 3, 64, 64), (2, 5, 64, 64)),
]

# Test cases for individual components
CNN_PATH_TEST_CASES = [
    (3, 1, (32, 64, 128), (1, 1, 64, 64, 64)),
    (2, 3, (64, 128, 256), (1, 3, 128, 128)),
]

TRANSFORMER_PATH_TEST_CASES = [
    (3, 1, 256, 8, 4, 8, (1, 1, 64, 64, 64)),
    (2, 3, 128, 4, 2, 16, (1, 3, 128, 128)),
]

FUSION_TEST_CASES = [
    (3, 256, 8, (1, 256, 8, 8, 8), (1, 256, 4, 4, 4)),
    (2, 128, 4, (1, 128, 16, 16), (1, 128, 8, 8)),
]


class TestMAGNUS(unittest.TestCase):
    """Test cases for MAGNUS model."""

    @parameterized.expand(MAGNUS_TEST_CASES)
    def test_magnus_shape(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        input_shape: tuple,
        expected_shape: tuple,
    ):
        """Test MAGNUS output shape."""
        model = MAGNUS(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(32, 64, 128, 256),  # Smaller for testing
            vit_depth=2,
            vit_patch_size=8,
        )
        model.eval()

        x = torch.randn(*input_shape)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, expected_shape)

    def test_magnus_deep_supervision(self):
        """Test MAGNUS with deep supervision."""
        model = MAGNUS(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 64, 128, 256),
            vit_depth=2,
            vit_patch_size=8,
            deep_supervision=True,
        )
        model.eval()

        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            main_out, aux_outs = model(x)

        self.assertEqual(main_out.shape, (1, 2, 32, 32, 32))
        self.assertEqual(len(aux_outs), 3)  # 4 stages - 1 = 3 aux outputs
        for aux_out in aux_outs:
            self.assertEqual(aux_out.shape, (1, 2, 32, 32, 32))

    def test_magnus_different_norms(self):
        """Test MAGNUS with different normalization types."""
        norms = [
            "batch",
            "instance",
            ("group", {"num_groups": 8}),  # GroupNorm requires num_groups
        ]
        for norm in norms:
            model = MAGNUS(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                features=(32, 64),
                vit_depth=1,
                vit_patch_size=8,
                norm=norm,
            )
            model.eval()

            x = torch.randn(1, 1, 32, 32, 32)
            with torch.no_grad():
                y = model(x)

            self.assertEqual(y.shape, (1, 2, 32, 32, 32))

    def test_magnus_gradient_flow(self):
        """Test gradient flow through MAGNUS."""
        model = MAGNUS(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 64),
            vit_depth=1,
            vit_patch_size=8,
        )
        model.train()

        x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_magnus_invalid_spatial_dims(self):
        """Test MAGNUS raises error for invalid spatial_dims."""
        with self.assertRaises(ValueError):
            MAGNUS(spatial_dims=4, in_channels=1, out_channels=2)


class TestCNNPath(unittest.TestCase):
    """Test cases for CNNPath."""

    @parameterized.expand(CNN_PATH_TEST_CASES)
    def test_cnn_path_shape(
        self,
        spatial_dims: int,
        in_channels: int,
        features: tuple,
        input_shape: tuple,
    ):
        """Test CNNPath output shapes."""
        model = CNNPath(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            features=features,
        )
        model.eval()

        x = torch.randn(*input_shape)
        with torch.no_grad():
            outputs = model(x)

        self.assertEqual(len(outputs), len(features))
        for i, (feat, out) in enumerate(zip(features, outputs)):
            self.assertEqual(out.shape[1], feat)
            # Each stage downsamples by factor of 2
            expected_spatial = [s // (2 ** (i + 1)) for s in input_shape[2:]]
            self.assertEqual(list(out.shape[2:]), expected_spatial)


class TestTransformerPath(unittest.TestCase):
    """Test cases for TransformerPath."""

    @parameterized.expand(TRANSFORMER_PATH_TEST_CASES)
    def test_transformer_path_shape(
        self,
        spatial_dims: int,
        in_channels: int,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        patch_size: int,
        input_shape: tuple,
    ):
        """Test TransformerPath output shape."""
        model = TransformerPath(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            patch_size=patch_size,
        )
        model.eval()

        x = torch.randn(*input_shape)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape[0], input_shape[0])  # Batch
        self.assertEqual(y.shape[1], hidden_dim)  # Channels
        expected_spatial = [s // patch_size for s in input_shape[2:]]
        self.assertEqual(list(y.shape[2:]), expected_spatial)


class TestCrossModalAttentionFusion(unittest.TestCase):
    """Test cases for CrossModalAttentionFusion."""

    @parameterized.expand(FUSION_TEST_CASES)
    def test_fusion_shape(
        self,
        spatial_dims: int,
        channels: int,
        num_heads: int,
        cnn_shape: tuple,
        vit_shape: tuple,
    ):
        """Test CrossModalAttentionFusion output shape."""
        model = CrossModalAttentionFusion(
            spatial_dims=spatial_dims,
            channels=channels,
            num_heads=num_heads,
        )
        model.eval()

        cnn_feat = torch.randn(*cnn_shape)
        vit_feat = torch.randn(*vit_shape)

        with torch.no_grad():
            y = model(cnn_feat, vit_feat)

        # Output should match CNN feature shape
        self.assertEqual(y.shape, cnn_shape)

    def test_fusion_invalid_channels(self):
        """Test fusion raises error when channels not divisible by heads."""
        with self.assertRaises(ValueError):
            CrossModalAttentionFusion(
                spatial_dims=3,
                channels=100,
                num_heads=8,  # 100 % 8 != 0
            )


class TestScaleAdaptiveConv(unittest.TestCase):
    """Test cases for ScaleAdaptiveConv."""

    def test_scale_adaptive_conv_3d(self):
        """Test ScaleAdaptiveConv 3D output shape."""
        model = ScaleAdaptiveConv(
            spatial_dims=3,
            in_channels=64,
            out_channels=128,
            kernel_sizes=(3, 5, 7),
        )
        model.eval()

        x = torch.randn(1, 64, 16, 16, 16)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, (1, 128, 16, 16, 16))

    def test_scale_adaptive_conv_2d(self):
        """Test ScaleAdaptiveConv 2D output shape."""
        model = ScaleAdaptiveConv(
            spatial_dims=2,
            in_channels=32,
            out_channels=64,
            kernel_sizes=(3, 5),
        )
        model.eval()

        x = torch.randn(1, 32, 32, 32)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, (1, 64, 32, 32))


class TestSEBlock(unittest.TestCase):
    """Test cases for SEBlock."""

    def test_se_block_3d(self):
        """Test SEBlock 3D output shape."""
        model = SEBlock(spatial_dims=3, channels=64, reduction=16)
        model.eval()

        x = torch.randn(1, 64, 8, 8, 8)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, x.shape)

    def test_se_block_2d(self):
        """Test SEBlock 2D output shape."""
        model = SEBlock(spatial_dims=2, channels=128, reduction=8)
        model.eval()

        x = torch.randn(2, 128, 16, 16)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, x.shape)

    def test_se_block_minimum_reduction(self):
        """Test SEBlock with small channel count."""
        # Reduction should be at least 1
        model = SEBlock(spatial_dims=2, channels=4, reduction=16)
        model.eval()

        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, x.shape)


class TestDecoderBlock(unittest.TestCase):
    """Test cases for DecoderBlock."""

    def test_decoder_block_3d(self):
        """Test DecoderBlock 3D output shape."""
        model = DecoderBlock(
            spatial_dims=3,
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )
        model.eval()

        x = torch.randn(1, 128, 8, 8, 8)
        skip = torch.randn(1, 64, 16, 16, 16)
        with torch.no_grad():
            y = model(x, skip)

        self.assertEqual(y.shape, (1, 64, 16, 16, 16))

    def test_decoder_block_2d(self):
        """Test DecoderBlock 2D output shape."""
        model = DecoderBlock(
            spatial_dims=2,
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            use_se=True,
        )
        model.eval()

        x = torch.randn(1, 256, 8, 8)
        skip = torch.randn(1, 128, 16, 16)
        with torch.no_grad():
            y = model(x, skip)

        self.assertEqual(y.shape, (1, 128, 16, 16))

    def test_decoder_block_no_se(self):
        """Test DecoderBlock without SE block."""
        model = DecoderBlock(
            spatial_dims=3,
            in_channels=64,
            skip_channels=32,
            out_channels=32,
            use_se=False,
        )
        model.eval()

        x = torch.randn(1, 64, 4, 4, 4)
        skip = torch.randn(1, 32, 8, 8, 8)
        with torch.no_grad():
            y = model(x, skip)

        self.assertEqual(y.shape, (1, 32, 8, 8, 8))


class TestMAGNUSMemory(unittest.TestCase):
    """Memory and performance tests for MAGNUS."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_magnus_cuda(self):
        """Test MAGNUS on CUDA."""
        model = MAGNUS(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 64, 128),
            vit_depth=2,
            vit_patch_size=8,
        ).cuda()
        model.eval()

        x = torch.randn(1, 1, 32, 32, 32, device="cuda")
        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.device.type, "cuda")
        self.assertEqual(y.shape, (1, 2, 32, 32, 32))


if __name__ == "__main__":
    unittest.main()
