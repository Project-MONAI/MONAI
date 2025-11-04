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

from monai.transforms.post.array import GenerateHeatmap
from tests.test_utils import TEST_NDARRAYS


def _argmax_nd(x) -> np.ndarray:
    """argmax for N-D array → returns coordinate vector (z,y,x) or (y,x)."""
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return np.asarray(np.unravel_index(np.argmax(x), x.shape))


# Test cases for 2D array inputs with different data types
TEST_CASES_2D = [
    [
        f"2d_basic_type{idx}",
        p(np.array([[4.2, 7.8], [12.3, 3.6]], dtype=np.float32)),
        {"sigma": 1.5, "spatial_shape": (16, 16)},
        (2, 16, 16),
    ]
    for idx, p in enumerate(TEST_NDARRAYS)
]

# Test cases for 3D torch outputs with explicit dtype
TEST_CASES_3D_TORCH = [
    [
        f"3d_torch_{str(dtype).replace('torch.', '')}",
        torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32),
        {"sigma": 1.0, "spatial_shape": (8, 8, 8), "dtype": dtype},
        (1, 8, 8, 8),
        dtype,
    ]
    for dtype in [torch.float32, torch.float64]
]

# Test cases for 3D numpy outputs with explicit dtype
TEST_CASES_3D_NUMPY = [
    [
        f"3d_numpy_{dtype_obj.__name__}",
        np.array([[1.5, 2.5, 3.5]], dtype=np.float32),
        {"sigma": 1.0, "spatial_shape": (8, 8, 8), "dtype": dtype_obj},
        (1, 8, 8, 8),
        dtype_obj,
    ]
    for dtype_obj in [np.float32, np.float64]
]

# Test cases for different sigma values
TEST_CASES_SIGMA = [
    [
        f"sigma_{sigma}",
        np.array([[8.0, 8.0]], dtype=np.float32),
        {"sigma": sigma, "spatial_shape": (16, 16)},
        (1, 16, 16),
    ]
    for sigma in [0.5, 1.0, 2.0, 3.0]
]

# Test cases for truncated parameter
TEST_CASES_TRUNCATED = [
    [
        f"truncated_{truncated}",
        np.array([[8.0, 8.0]], dtype=np.float32),
        {"sigma": 2.0, "spatial_shape": (32, 32), "truncated": truncated},
        (1, 32, 32),
    ]
    for truncated in [2.0, 4.0, 6.0]
]

# Test cases for device and dtype propagation (torch only)
test_device = "cuda:0" if torch.cuda.is_available() else "cpu"
test_dtypes = [torch.float32, torch.float64]
if torch.cuda.is_available():
    test_dtypes.append(torch.float16)

TEST_CASES_DEVICE_DTYPE = [
    [
        f"{test_device.split(':')[0]}_{str(dtype).replace('torch.', '')}",
        torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32, device=test_device),
        {"sigma": 1.2, "spatial_shape": (10, 10, 10), "dtype": dtype},
        (1, 10, 10, 10),
        dtype,
        test_device,
    ]
    for dtype in test_dtypes
]


class TestGenerateHeatmap(unittest.TestCase):
    @parameterized.expand(TEST_CASES_2D)
    def test_array_2d(self, _, points, params, expected_shape):
        transform = GenerateHeatmap(**params)
        heatmap = transform(points)

        # Check output type matches input type
        if isinstance(points, torch.Tensor):
            self.assertIsInstance(heatmap, torch.Tensor)
            self.assertEqual(heatmap.dtype, torch.float32)  # Default dtype for torch
            heatmap_np = heatmap.cpu().numpy()
            points_np = points.cpu().numpy()
        else:
            self.assertIsInstance(heatmap, np.ndarray)
            self.assertEqual(heatmap.dtype, np.float32)  # Default dtype for numpy
            heatmap_np = heatmap
            points_np = points

        self.assertEqual(heatmap.shape, expected_shape)
        np.testing.assert_allclose(heatmap_np.max(axis=(1, 2)), np.ones(expected_shape[0]), rtol=1e-5, atol=1e-5)

        # peak should be close to original point location (<= 1px tolerance due to discretization)
        for idx in range(expected_shape[0]):
            peak = _argmax_nd(heatmap_np[idx])
            self.assertTrue(np.all(np.abs(peak - points_np[idx]) <= 1.0), msg=f"peak={peak}, point={points_np[idx]}")
            self.assertLess(heatmap_np[idx, 0, 0], 1e-3)

    @parameterized.expand(TEST_CASES_3D_TORCH)
    def test_array_3d_torch_output(self, _, points, params, expected_shape, expected_dtype):
        transform = GenerateHeatmap(**params)
        heatmap = transform(points)

        self.assertIsInstance(heatmap, torch.Tensor)
        self.assertEqual(heatmap.device, points.device)
        self.assertEqual(tuple(heatmap.shape), expected_shape)
        self.assertEqual(heatmap.dtype, expected_dtype)
        self.assertTrue(torch.isclose(heatmap.max(), torch.tensor(1.0, dtype=heatmap.dtype, device=heatmap.device)))

    @parameterized.expand(TEST_CASES_3D_NUMPY)
    def test_array_3d_numpy_output(self, _, points, params, expected_shape, expected_dtype):
        transform = GenerateHeatmap(**params)
        heatmap = transform(points)

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, expected_shape)
        self.assertEqual(heatmap.dtype, expected_dtype)
        np.testing.assert_allclose(heatmap.max(), 1.0, rtol=1e-5)

    @parameterized.expand(TEST_CASES_DEVICE_DTYPE)
    def test_array_torch_device_and_dtype_propagation(
        self, _, pts, params, expected_shape, expected_dtype, expected_device
    ):
        tr = GenerateHeatmap(**params)
        hm = tr(pts)

        self.assertIsInstance(hm, torch.Tensor)
        self.assertEqual(str(hm.device).split(":")[0], expected_device.split(":")[0])
        self.assertEqual(hm.dtype, expected_dtype)
        self.assertEqual(tuple(hm.shape), expected_shape)
        self.assertTrue(torch.all(hm >= 0))

    def test_array_channel_order_identity(self):
        # ensure the order of channels follows the order of input points
        pts = np.array([[2.0, 2.0], [12.0, 2.0], [2.0, 12.0]], dtype=np.float32)  # point A  # point B  # point C
        hm = GenerateHeatmap(sigma=1.2, spatial_shape=(16, 16))(pts)

        self.assertIsInstance(hm, np.ndarray)
        self.assertEqual(hm.shape, (3, 16, 16))

        peaks = np.vstack([_argmax_nd(hm[i]) for i in range(3)])
        # y,x close to points
        np.testing.assert_allclose(peaks, pts, atol=1.0)

    def test_array_points_out_of_bounds(self):
        # points outside spatial domain: heatmap should still be valid (no NaN/Inf) and not all-zeros
        pts = np.array(
            [[-5.0, -5.0], [100.0, 100.0], [8.0, 8.0]],  # outside top-left  # outside bottom-right  # inside
            dtype=np.float32,
        )
        hm = GenerateHeatmap(sigma=2.0, spatial_shape=(16, 16))(pts)

        self.assertIsInstance(hm, np.ndarray)
        self.assertEqual(hm.shape, (3, 16, 16))
        self.assertFalse(np.isnan(hm).any() or np.isinf(hm).any())

        # inside point channel should have max≈1; others may clip at border (≤1)
        self.assertGreater(hm[2].max(), 0.9)

    @parameterized.expand(TEST_CASES_SIGMA)
    def test_array_sigma_scaling_effect(self, _, pt, params, expected_shape):
        heatmap = GenerateHeatmap(**params)(pt)[0]
        self.assertEqual(heatmap.shape, expected_shape[1:])

        # All should have peak normalized to 1.0
        np.testing.assert_allclose(heatmap.max(), 1.0, rtol=1e-5)

        # Verify heatmap is valid
        self.assertFalse(np.isnan(heatmap).any() or np.isinf(heatmap).any())

    def test_invalid_points_shape_raises(self):
        # points must be (N, D) with D in {2,3}
        tr = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8))
        with self.assertRaises((ValueError, AssertionError, IndexError, RuntimeError)):
            tr(np.zeros((2,), dtype=np.float32))  # wrong rank

        with self.assertRaises((ValueError, AssertionError, IndexError, RuntimeError)):
            tr(np.zeros((2, 4), dtype=np.float32))  # D=4 unsupported

    @parameterized.expand(TEST_CASES_TRUNCATED)
    def test_truncated_parameter(self, _, pt, params, expected_shape):
        heatmap = GenerateHeatmap(**params)(pt)[0]

        # All should have same peak value (normalized to 1.0)
        np.testing.assert_allclose(heatmap.max(), 1.0, rtol=1e-5)

        # Verify shape and no NaN/Inf
        self.assertEqual(heatmap.shape, expected_shape[1:])
        self.assertFalse(np.isnan(heatmap).any() or np.isinf(heatmap).any())

    def test_torch_to_torch_type_preservation(self):
        """Test that torch input produces torch output"""
        pts = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
        hm = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8))(pts)

        self.assertIsInstance(hm, torch.Tensor)
        self.assertEqual(hm.dtype, torch.float32)
        self.assertEqual(hm.device, pts.device)

    def test_numpy_to_numpy_type_preservation(self):
        """Test that numpy input produces numpy output"""
        pts = np.array([[4.0, 4.0]], dtype=np.float32)
        hm = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8))(pts)

        self.assertIsInstance(hm, np.ndarray)
        self.assertEqual(hm.dtype, np.float32)

    def test_dtype_override_torch(self):
        """Test dtype parameter works with torch tensors"""
        pts = torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float32)
        hm = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8, 8), dtype=torch.float64)(pts)

        self.assertIsInstance(hm, torch.Tensor)
        self.assertEqual(hm.dtype, torch.float64)

    def test_dtype_override_numpy(self):
        """Test dtype parameter works with numpy arrays"""
        pts = np.array([[4.0, 4.0, 4.0]], dtype=np.float32)
        hm = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8, 8), dtype=np.float64)(pts)

        self.assertIsInstance(hm, np.ndarray)
        self.assertEqual(hm.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
