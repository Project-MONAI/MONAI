# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest
import math
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms.post.array import GenerateHeatmap
from monai.transforms.post.dictionary import GenerateHeatmapd
from tests.test_utils import assert_allclose


def _argmax_nd(x: np.ndarray) -> np.ndarray:
    """argmax for N-D array → returns coordinate vector (z,y,x) or (y,x)."""
    return np.asarray(np.unravel_index(np.argmax(x), x.shape))


class TestGenerateHeatmap(unittest.TestCase):
    def test_array_2d(self):
        points = np.array([[4.2, 7.8], [12.3, 3.6]], dtype=np.float32)
        transform = GenerateHeatmap(sigma=1.5, spatial_shape=(16, 16))

        heatmap = transform(points)

        self.assertEqual(heatmap.shape, (2, 16, 16))
        self.assertEqual(heatmap.dtype, np.float32)
        np.testing.assert_allclose(heatmap.max(axis=(1, 2)), np.ones(2), rtol=1e-5, atol=1e-5)

        # peak should be close to original point location (<= 1px tolerance due to discretization)
        for idx, channel in enumerate(heatmap):
            peak = _argmax_nd(channel)
            self.assertTrue(np.all(np.abs(peak - points[idx]) <= 1.0), msg=f"peak={peak}, point={points[idx]}")
            self.assertLess(channel[0, 0], 1e-3)

    def test_array_3d_torch_output(self):
        points = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
        transform = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8, 8), dtype=torch.float32)

        heatmap = transform(points.to(device=points.device))

        self.assertIsInstance(heatmap, torch.Tensor)
        self.assertEqual(heatmap.device, points.device)
        self.assertEqual(tuple(heatmap.shape), (1, 8, 8, 8))
        self.assertTrue(torch.isclose(heatmap.max(), torch.tensor(1.0, dtype=heatmap.dtype, device=heatmap.device)))

    def test_array_torch_device_and_dtype_propagation(self):
        # verify dtype parameter honored and CUDA (if available)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        pts = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32, device=device)
        tr = GenerateHeatmap(sigma=1.2, spatial_shape=(10, 10, 10), dtype=dtype)

        hm = tr(pts)
        self.assertIsInstance(hm, torch.Tensor)
        self.assertEqual(hm.device, device)
        self.assertEqual(hm.dtype, dtype)
        self.assertEqual(tuple(hm.shape), (1, 10, 10, 10))
        self.assertTrue(torch.all(hm >= 0))

    def test_array_channel_order_identity(self):
        # ensure the order of channels follows the order of input points
        pts = np.array(
            [
                [2.0, 2.0],  # point A
                [12.0, 2.0],  # point B
                [2.0, 12.0],  # point C
            ],
            dtype=np.float32,
        )
        hm = GenerateHeatmap(sigma=1.2, spatial_shape=(16, 16))(pts)
        self.assertEqual(hm.shape, (3, 16, 16))

        peaks = np.vstack([_argmax_nd(hm[i]) for i in range(3)])
        # y,x close to points
        np.testing.assert_allclose(peaks, pts, atol=1.0)

    def test_array_points_out_of_bounds(self):
        # points outside spatial domain: heatmap should still be valid (no NaN/Inf) and not all-zeros
        pts = np.array(
            [
                [-5.0, -5.0],  # outside top-left
                [100.0, 100.0],  # outside bottom-right
                [8.0, 8.0],  # inside
            ],
            dtype=np.float32,
        )
        hm = GenerateHeatmap(sigma=2.0, spatial_shape=(16, 16))(pts)
        self.assertEqual(hm.shape, (3, 16, 16))
        self.assertFalse(np.isnan(hm).any() or np.isinf(hm).any())

        # inside point channel should have max≈1; others may clip at border (≤1)
        self.assertGreater(hm[2].max(), 0.9)

    def test_array_sigma_scaling_effect(self):
        # Larger sigma should spread mass (lower peak), smaller sigma higher peak
        pt = np.array([[8.0, 8.0]], dtype=np.float32)
        small = GenerateHeatmap(sigma=0.8, spatial_shape=(16, 16))(pt)[0]
        large = GenerateHeatmap(sigma=2.5, spatial_shape=(16, 16))(pt)[0]
        self.assertGreater(small.max(), large.max() - 1e-6)  # small sigma peak >= large sigma peak

    def test_dict_with_reference_meta(self):
        points = np.array([[2.5, 2.5, 3.0], [5.0, 5.0, 4.0]], dtype=np.float32)
        affine = torch.eye(4)
        image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=affine)
        image.meta["spatial_shape"] = (8, 8, 8)
        data = {"points": points, "image": image}

        transform = GenerateHeatmapd(
            keys="points",
            heatmap_keys="heatmap",
            ref_image_keys="image",
            sigma=2.0,
        )

        result = transform(data)
        heatmap = result["heatmap"]

        self.assertIsInstance(heatmap, MetaTensor)
        self.assertEqual(tuple(heatmap.shape), (2, 8, 8, 8))
        self.assertEqual(heatmap.meta["spatial_shape"], (8, 8, 8))
        assert_allclose(heatmap.affine, image.affine, type_test=False)
        np.testing.assert_allclose(heatmap.cpu().numpy().max(axis=(1, 2, 3)), np.ones(2), rtol=1e-5, atol=1e-5)

    def test_dict_static_shape(self):
        points = np.array([[1.0, 1.0]], dtype=np.float32)
        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap", spatial_shape=(6, 6))

        result = transform({"points": points})
        heatmap = result["heatmap"]
        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, (1, 6, 6))

    def test_dict_missing_shape_raises(self):
        # Without ref image or explicit spatial_shape, must raise
        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap")
        with self.assertRaises(ValueError):
            transform({"points": np.zeros((1, 2), dtype=np.float32)})

    def test_invalid_points_shape_raises(self):
        # points must be (N, D) with D in {2,3}
        tr = GenerateHeatmap(sigma=1.0, spatial_shape=(8, 8))
        with self.assertRaises((ValueError, AssertionError, IndexError, RuntimeError)):
            tr(np.zeros((2,), dtype=np.float32))  # wrong rank

        with self.assertRaises((ValueError, AssertionError, IndexError, RuntimeError)):
            tr(np.zeros((2, 4), dtype=np.float32))  # D=4 unsupported

    def test_dict_dtype_control(self):
        # Ensure dtype argument controls output dtype for dictionary transform too
        points = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        ref = MetaTensor(torch.zeros((1, 10, 10, 10), dtype=torch.float32), affine=torch.eye(4))
        d = {"pts": points, "img": ref}

        tr = GenerateHeatmapd(keys="pts", heatmap_keys="hm", ref_image_keys="img", sigma=1.4, dtype=torch.float16)
        out = tr(d)
        hm = out["hm"]
        self.assertIsInstance(hm, MetaTensor)
        self.assertEqual(tuple(hm.shape), (1, 10, 10, 10))
        self.assertEqual(hm.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
