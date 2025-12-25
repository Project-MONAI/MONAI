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
Tests for the 3D Radial Fourier Transform.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import RadialFourier3D, RadialFourierFeatures3D
from monai.utils import set_determinism


class TestRadialFourier3D(unittest.TestCase):
    """Test cases for RadialFourier3D transform."""

    def setUp(self):
        """Set up test fixtures."""
        set_determinism(seed=42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create test data
        self.test_image_3d = torch.randn(1, 32, 64, 64, device=self.device)  # Batch, D, H, W

    def tearDown(self):
        """Clean up after tests."""
        set_determinism(seed=None)

    @parameterized.expand(
        [
            [{"radial_bins": 32, "return_magnitude": True}, (1, 32)],
            [{"radial_bins": 64, "return_magnitude": True, "return_phase": True}, (1, 128)],
            [{"radial_bins": None, "return_magnitude": True}, (1, 32, 64, 64)],
            [{"radial_bins": 16, "return_magnitude": True, "max_frequency": 0.5}, (1, 16)],
        ]
    )
    def test_output_shape(self, params, expected_shape):
        """Test that output shape matches expectations."""
        transform = RadialFourier3D(**params)
        result = transform(self.test_image_3d)
        self.assertEqual(result.shape, expected_shape)

    def test_complex_input(self):
        """Test with complex-valued input."""
        complex_image = torch.complex(
            torch.randn(1, 32, 64, 64, device=self.device),
            torch.randn(1, 32, 64, 64, device=self.device),
        )
        transform = RadialFourier3D(radial_bins=32, return_magnitude=True)
        result = transform(complex_image)
        self.assertEqual(result.shape, (1, 32))

    def test_normalization(self):
        """Test normalization affects output scale."""
        transform1 = RadialFourier3D(radial_bins=32, normalize=True)
        transform2 = RadialFourier3D(radial_bins=32, normalize=False)

        result1 = transform1(self.test_image_3d)
        result2 = transform2(self.test_image_3d)

        # Normalized result should be smaller
        self.assertLess(torch.abs(result1).mean().item(), torch.abs(result2).mean().item())

    def test_inverse_transform(self):
        """Test approximate inverse transform."""
        # Use full spectrum for invertibility
        transform = RadialFourier3D(radial_bins=None, normalize=True, return_magnitude=True, return_phase=True)

        # Forward transform
        spectrum = transform(self.test_image_3d)

        # Inverse transform
        reconstructed = transform.inverse(spectrum, self.test_image_3d.shape[-3:])

        # Should have same shape
        self.assertEqual(reconstructed.shape, self.test_image_3d.shape)

        # Should approximately reconstruct original
        self.assertTrue(torch.allclose(reconstructed, self.test_image_3d, atol=1e-5))

    def test_deterministic(self):
        """Test that transform is deterministic."""
        transform = RadialFourier3D(radial_bins=32)

        result1 = transform(self.test_image_3d)
        result2 = transform(self.test_image_3d)

        self.assertTrue(torch.allclose(result1, result2, rtol=1e-5))

    def test_numpy_input(self):
        """Test that numpy arrays are accepted."""
        np_image = self.test_image_3d.cpu().numpy()
        transform = RadialFourier3D(radial_bins=32)

        result = transform(np_image)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 32))

    @parameterized.expand(
        [
            [{"max_frequency": -0.1}],  # Invalid negative
            [{"max_frequency": 1.5}],  # Invalid > 1.0
            [{"radial_bins": 0}],  # Invalid zero bins
            [{"return_magnitude": False, "return_phase": False}],  # No output requested
        ]
    )
    def test_invalid_parameters(self, params):
        """Test that invalid parameters raise errors."""
        with self.assertRaises(ValueError):
            RadialFourier3D(**params)

    def test_spatial_dims_parameter(self):
        """Test custom spatial dimensions."""
        # Test with 4D input but spatial dims in middle
        image = torch.randn(2, 32, 64, 64, 3, device=self.device)  # Batch, D, H, W, Channels
        transform = RadialFourier3D(radial_bins=16, spatial_dims=(1, 2, 3))
        result = transform(image)
        self.assertEqual(result.shape, (2, 3, 16))

    def test_batch_processing(self):
        """Test processing batch of images."""
        batch_size = 4
        batch_image = torch.randn(batch_size, 32, 64, 64, device=self.device)
        transform = RadialFourier3D(radial_bins=32)
        result = transform(batch_image)
        self.assertEqual(result.shape, (batch_size, 32))


class TestRadialFourierFeatures3D(unittest.TestCase):
    """Test cases for RadialFourierFeatures3D transform."""

    def setUp(self):
        """Set up test fixtures."""
        set_determinism(seed=42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_image = torch.randn(2, 32, 64, 64, device=self.device)

    def tearDown(self):
        """Clean up after tests."""
        set_determinism(seed=None)

    def test_feature_extraction(self):
        """Test multi-scale feature extraction."""
        transform = RadialFourierFeatures3D(n_bins_list=[16, 32, 64], return_types=["magnitude"])

        features = transform(self.test_image)
        expected_features = 16 + 32 + 64  # Sum of all bins

        self.assertEqual(features.shape, (2, expected_features))

    def test_multiple_return_types(self):
        """Test with multiple return types."""
        transform = RadialFourierFeatures3D(n_bins_list=[16, 32], return_types=["magnitude", "phase"])

        features = transform(self.test_image)
        # Each bin count appears twice (magnitude and phase)
        expected_features = (16 + 32) * 2

        self.assertEqual(features.shape, (2, expected_features))

    def test_complex_output(self):
        """Test complex output type."""
        transform = RadialFourierFeatures3D(n_bins_list=[16], return_types=["complex"])

        features = transform(self.test_image)
        # Complex returns both magnitude and phase concatenated
        self.assertEqual(features.shape, (2, 16 * 2))

    def test_empty_bins_list(self):
        """Test with empty bins list raises ValueError."""
        with self.assertRaises(ValueError):
            RadialFourierFeatures3D(n_bins_list=[], return_types=["magnitude"])

    def test_numpy_compatibility(self):
        """Test with numpy input."""
        np_image = self.test_image.cpu().numpy()
        transform = RadialFourierFeatures3D(n_bins_list=[16, 32])

        features = transform(np_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (2, 16 + 32))


if __name__ == "__main__":
    unittest.main()
