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


def _peak_coord(channel: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(channel)
    return torch.stack(torch.unravel_index(idx, channel.shape))


# Test cases for dictionary transforms with reference image
# Only test with non-MetaTensor types to avoid affine conflicts
TEST_CASES_WITH_REF = [
    [
        "dict_with_ref_3d_numpy",
        np.array([[2.5, 2.5, 3.0], [5.0, 5.0, 4.0]], dtype=np.float32),
        {"sigma": 2.0},
        (2, 8, 8, 8),
        torch.float32,
        True,  # uses reference image
    ],
    [
        "dict_with_ref_3d_torch",
        torch.tensor([[2.5, 2.5, 3.0], [5.0, 5.0, 4.0]], dtype=torch.float32),
        {"sigma": 2.0},
        (2, 8, 8, 8),
        torch.float32,
        True,  # uses reference image
    ],
]

# Test cases for dictionary transforms with static spatial shape
TEST_CASES_STATIC_SHAPE = [
    [
        f"dict_static_shape_{len(shape)}d",
        np.array([[1.0] * len(shape)], dtype=np.float32),
        {"spatial_shape": shape},
        (1, *shape),
        np.float32,
    ]
    for shape in [(6, 6), (8, 8, 8), (10, 10, 10)]
]

# Test cases for dtype control
TEST_CASES_DTYPE = [
    [
        f"dict_dtype_{str(dtype).replace('torch.', '')}",
        np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
        {"sigma": 1.4, "dtype": dtype},
        (1, 10, 10, 10),
        dtype,
    ]
    for dtype in [torch.float16, torch.float32, torch.float64]
]

# Test cases for various sigma values
TEST_CASES_SIGMA_VALUES = [
    [
        f"dict_sigma_{sigma}",
        np.array([[4.0, 4.0, 4.0]], dtype=np.float32),
        {"sigma": sigma, "spatial_shape": (8, 8, 8)},
        (1, 8, 8, 8),
    ]
    for sigma in [0.5, 1.0, 2.0, 3.0]
]


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

        # Verify no NaN or Inf values
        self.assertFalse(np.isnan(heatmap).any() or np.isinf(heatmap).any())

        # Verify max value is 1.0 for normalized heatmaps
        np.testing.assert_allclose(heatmap.max(), 1.0, rtol=1e-5)

    def test_dict_missing_shape_raises(self):
        # Without ref image or explicit spatial_shape, must raise
        transform = GenerateHeatmapd(keys="points", heatmap_keys="heatmap")
        with self.assertRaisesRegex(ValueError, "spatial_shape|ref_image_keys"):
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

    def test_dict_mismatched_heatmap_keys_length(self):
        """Test ValueError when heatmap_keys length doesn't match keys"""
        with self.assertRaises(ValueError):
            GenerateHeatmapd(
                keys=["pts1", "pts2"],
                heatmap_keys=["hm1", "hm2", "hm3"],  # Mismatch: 3 heatmap keys for 2 input keys
                spatial_shape=(8, 8),
            )

    def test_dict_mismatched_ref_image_keys_length(self):
        """Test ValueError when ref_image_keys length doesn't match keys"""
        with self.assertRaises(ValueError):
            GenerateHeatmapd(
                keys=["pts1", "pts2"],
                heatmap_keys=["hm1", "hm2"],
                ref_image_keys=["img1", "img2", "img3"],  # Mismatch: 3 ref keys for 2 input keys
                spatial_shape=(8, 8),
            )

    def test_dict_per_key_spatial_shape_mismatch(self):
        """Test ValueError when per-key spatial_shape length doesn't match keys"""
        with self.assertRaises(ValueError):
            GenerateHeatmapd(
                keys=["pts1", "pts2"],
                heatmap_keys=["hm1", "hm2"],
                spatial_shape=[(8, 8), (8, 8), (8, 8)],  # Mismatch: 3 shapes for 2 keys
                sigma=1.0,
            )

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

        # Heatmap should inherit affine from the reference image
        assert_allclose(heatmap.affine, image.affine, type_test=False)

    def test_world_points_with_reference_affine_and_visibility(self):
        affine = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))
        image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=affine)
        image.meta["spatial_shape"] = (8, 8, 8)
        points = torch.tensor(
            [
                [4.0, 6.0, 8.0],  # voxel coordinate [2, 3, 4]
                [20.0, 0.0, 0.0],  # out of bounds after affine conversion
                [float("nan"), 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        transform = GenerateHeatmapd(
            keys="points",
            heatmap_keys="heatmap",
            ref_image_keys="image",
            coordinate_space="world",
            visibility_keys="visible",
            sigma=1.0,
        )
        result = transform({"points": points, "image": image})

        heatmap = result["heatmap"]
        self.assertIsInstance(heatmap, MetaTensor)
        self.assertEqual(tuple(heatmap.shape), (3, 8, 8, 8))
        assert_allclose(_peak_coord(heatmap[0]), torch.tensor([2, 3, 4]), type_test=False)
        self.assertTrue(torch.equal(result["visible"], torch.tensor([True, False, False])))
        self.assertGreater(heatmap[0].max(), 0.99)
        self.assertEqual(float(heatmap[1].max()), 0.0)
        self.assertEqual(float(heatmap[2].max()), 0.0)

    def test_world_points_with_translated_rotated_affine(self):
        affine = torch.tensor(
            [[0.0, -2.0, 0.0, 10.0], [3.0, 0.0, 0.0, 20.0], [0.0, 0.0, 4.0, 30.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=affine)
        image.meta["spatial_shape"] = (8, 8, 8)
        voxel_point = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        world_point = affine[:3, :3] @ voxel_point + affine[:3, 3]

        transform = GenerateHeatmapd(
            keys="points",
            heatmap_keys="heatmap",
            ref_image_keys="image",
            coordinate_space="world",
            visibility_keys="visible",
            sigma=1.0,
        )
        result = transform({"points": world_point[None], "image": image})

        assert_allclose(_peak_coord(result["heatmap"][0]), voxel_point.to(torch.long), type_test=False)
        self.assertTrue(torch.equal(result["visible"], torch.tensor([True])))

    def test_world_metatensor_points_use_point_affine(self):
        image = MetaTensor(torch.zeros((1, 8, 8, 8), dtype=torch.float32), affine=torch.eye(4))
        image.meta["spatial_shape"] = (8, 8, 8)
        points_affine = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))
        points = MetaTensor(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), affine=points_affine)

        transform = GenerateHeatmapd(
            keys="points",
            heatmap_keys="heatmap",
            ref_image_keys="image",
            coordinate_space="world",
            visibility_keys="visible",
            sigma=1.0,
        )
        result = transform({"points": points, "image": image})

        assert_allclose(_peak_coord(result["heatmap"][0]), torch.tensor([2, 4, 6]), type_test=False)
        self.assertIsInstance(result["visible"], torch.Tensor)
        self.assertNotIsInstance(result["visible"], MetaTensor)
        self.assertTrue(bool(result["visible"][0]))

    def test_world_points_require_reference_affine(self):
        transform = GenerateHeatmapd(
            keys="points", heatmap_keys="heatmap", spatial_shape=(8, 8, 8), coordinate_space="world"
        )
        with self.assertRaisesRegex(ValueError, "reference|affine|ref_image_keys"):
            transform({"points": torch.zeros((1, 3), dtype=torch.float32)})

    def test_invalid_coordinate_space_raises(self):
        with self.assertRaisesRegex(ValueError, "coordinate_space"):
            GenerateHeatmapd(keys="points", heatmap_keys="heatmap", spatial_shape=(8, 8), coordinate_space="scanner")

    def test_visibility_key_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            GenerateHeatmapd(
                keys=["pts1", "pts2"],
                heatmap_keys=["hm1", "hm2"],
                visibility_keys=["visible1", "visible2", "visible3"],
                spatial_shape=(8, 8),
            )


if __name__ == "__main__":
    unittest.main()
