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

from monai.data import MetaTensor
from monai.transforms.post.dictionary import GenerateHeatmapd
from tests.test_utils import assert_allclose

# Test cases for dictionary transforms with reference image
# Only test with non-MetaTensor types to avoid affine conflicts
TEST_CASES_WITH_REF = []
TEST_CASES_WITH_REF.append(
    [
        "dict_with_ref_3d_numpy",
        np.array([[2.5, 2.5, 3.0], [5.0, 5.0, 4.0]], dtype=np.float32),
        {"sigma": 2.0},
        (2, 8, 8, 8),
        torch.float32,
        True,  # uses reference image
    ]
)
TEST_CASES_WITH_REF.append(
    [
        "dict_with_ref_3d_torch",
        torch.tensor([[2.5, 2.5, 3.0], [5.0, 5.0, 4.0]], dtype=torch.float32),
        {"sigma": 2.0},
        (2, 8, 8, 8),
        torch.float32,
        True,  # uses reference image
    ]
)

# Test cases for dictionary transforms with static spatial shape
TEST_CASES_STATIC_SHAPE = []
for shape in [(6, 6), (8, 8, 8), (10, 10, 10)]:
    TEST_CASES_STATIC_SHAPE.append(
        [
            f"dict_static_shape_{len(shape)}d",
            np.array([[1.0] * len(shape)], dtype=np.float32),
            {"spatial_shape": shape},
            (1, *shape),
            np.float32,
        ]
    )

# Test cases for dtype control
TEST_CASES_DTYPE = []
for dtype in [torch.float16, torch.float32, torch.float64]:
    TEST_CASES_DTYPE.append(
        [
            f"dict_dtype_{str(dtype).replace('torch.', '')}",
            np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
            {"sigma": 1.4, "dtype": dtype},
            (1, 10, 10, 10),
            dtype,
        ]
    )

# Test cases for batched dictionary transforms
TEST_CASES_BATCHED = []
TEST_CASES_BATCHED.append(
    [
        "dict_batched_3d",
        torch.tensor([[[1.5, 2.5, 3.5]], [[4.5, 5.5, 6.5]]], dtype=torch.float32),
        {"sigma": 1.0},
        (2, 1, 8, 8, 8),
        torch.float32,
    ]
)

# Test cases for various sigma values
TEST_CASES_SIGMA_VALUES = []
for sigma in [0.5, 1.0, 2.0, 3.0]:
    TEST_CASES_SIGMA_VALUES.append(
        [
            f"dict_sigma_{sigma}",
            np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
            {"sigma": sigma, "spatial_shape": (8, 8, 8)},
            (1, 8, 8, 8),
        ]
    )


class TestGenerateHeatmapd(unittest.TestCase):
    @parameterized.expand(TEST_CASES_WITH_REF)
    def test_dict_with_reference_meta(self, _, points, params, expected_shape, *_unused):
        affine = torch.eye(4)
        image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=affine)
        image.meta["spatial_shape"] = (8, 8, 8)
        data = {"points": points, "image": image}

        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap", ref_image_keys="image", **params)
        result = transform(data)
        heatmap = result["heatmap"]

        self.assertIsInstance(heatmap, MetaTensor)
        self.assertEqual(tuple(heatmap.shape), expected_shape)
        self.assertEqual(heatmap.meta["spatial_shape"], (8, 8, 8))
        # The heatmap should inherit the reference image's affine
        assert_allclose(heatmap.affine, image.affine, type_test=False)

        # Check max values are normalized to 1.0
        max_vals = heatmap.cpu().numpy().max(axis=tuple(range(1, len(expected_shape))))
        np.testing.assert_allclose(max_vals, np.ones(expected_shape[0]), rtol=1e-5, atol=1e-5)

    @parameterized.expand(TEST_CASES_STATIC_SHAPE)
    def test_dict_static_shape(self, _, points, params, expected_shape, expected_dtype):
        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap", **params)
        result = transform({"points": points})
        heatmap = result["heatmap"]

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, expected_shape)
        self.assertEqual(heatmap.dtype, expected_dtype)

    def test_dict_missing_shape_raises(self):
        # Without ref image or explicit spatial_shape, must raise
        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap")
        with self.assertRaises(ValueError):
            transform({"points": np.zeros((1, 2), dtype=np.float32)})

    @parameterized.expand(TEST_CASES_DTYPE)
    def test_dict_dtype_control(self, _, points, params, expected_shape, expected_dtype):
        ref = MetaTensor(torch.zeros((1, 10, 10, 10), dtype=torch.float32), affine=torch.eye(4))
        d = {"pts": points, "img": ref}

        tr = GenerateHeatmapd(keys="pts", heatmap_keys="hm", ref_image_keys="img", **params)
        out = tr(d)
        hm = out["hm"]

        self.assertIsInstance(hm, MetaTensor)
        self.assertEqual(tuple(hm.shape), expected_shape)
        self.assertEqual(hm.dtype, expected_dtype)

    @parameterized.expand(TEST_CASES_BATCHED)
    def test_dict_batched_with_ref(self, _, points, params, expected_shape, _expected_dtype):
        affine = torch.eye(4)
        # A single reference image is used for the whole batch
        image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=affine)
        image.meta["spatial_shape"] = (8, 8, 8)
        data = {"points": points, "image": image}

        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap", ref_image_keys="image", **params)
        result = transform(data)
        heatmap = result["heatmap"]

        self.assertIsInstance(heatmap, MetaTensor)
        self.assertEqual(tuple(heatmap.shape), expected_shape)
        self.assertEqual(heatmap.meta["spatial_shape"], (8, 8, 8))
        assert_allclose(heatmap.affine, image.affine, type_test=False)

        # Check max values
        hm2 = heatmap.reshape(heatmap.shape[0], heatmap.shape[1], -1)
        max_vals = hm2.max(dim=2)[0]
        np.testing.assert_allclose(
            max_vals.cpu().numpy(), np.ones((expected_shape[0], expected_shape[1])), rtol=1e-5, atol=1e-5
        )

    @parameterized.expand(TEST_CASES_SIGMA_VALUES)
    def test_dict_various_sigma(self, _, points, params, expected_shape):
        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap", **params)
        result = transform({"points": points})
        heatmap = result["heatmap"]

        self.assertEqual(heatmap.shape, expected_shape)
        # Verify heatmap is normalized
        np.testing.assert_allclose(heatmap.max(), 1.0, rtol=1e-5)
        # Verify no NaN or Inf
        self.assertFalse(np.isnan(heatmap).any() or np.isinf(heatmap).any())

    def test_dict_multiple_keys(self):
        """Test dictionary transform with multiple input/output keys"""
        points1 = np.array([[2.0, 2.0]], dtype=np.float32)
        points2 = np.array([[4.0, 4.0]], dtype=np.float32)

        data = {"pts1": points1, "pts2": points2}
        transform = GenerateHeatmapd(
            keys=["pts1", "pts2"], heatmap_keys=["hm1", "hm2"], spatial_shape=(8, 8), sigma=1.0
        )

        result = transform(data)

        self.assertIn("hm1", result)
        self.assertIn("hm2", result)
        self.assertEqual(result["hm1"].shape, (1, 8, 8))
        self.assertEqual(result["hm2"].shape, (1, 8, 8))

        # Verify peaks are at different locations
        self.assertNotEqual(np.argmax(result["hm1"]), np.argmax(result["hm2"]))

    def test_metatensor_points_with_ref(self):
        """Test MetaTensor points with reference image - documents current behavior"""
        from monai.data import MetaTensor

        # Create MetaTensor points with non-identity affine
        points_affine = torch.tensor([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]])
        points = MetaTensor(torch.tensor([[2.5, 2.5, 3.0], [5.0, 5.0, 4.0]], dtype=torch.float32), affine=points_affine)

        # Reference image with identity affine
        ref_affine = torch.eye(4)
        image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=ref_affine)
        image.meta["spatial_shape"] = (8, 8, 8)

        data = {"points": points, "image": image}
        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap", ref_image_keys="image", sigma=2.0)
        result = transform(data)
        heatmap = result["heatmap"]

        self.assertIsInstance(heatmap, MetaTensor)
        self.assertEqual(tuple(heatmap.shape), (2, 8, 8, 8))

        # Note: Currently the heatmap may inherit affine from points MetaTensor
        # This test documents the current behavior
        # Ideally, the heatmap should use the reference image's affine


if __name__ == "__main__":
    unittest.main()
