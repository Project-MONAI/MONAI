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

from monai.data import MetaTensor
from monai.transforms import SpatialCropd
from monai.transforms.utility.dictionary import TransformPointsWorldToImaged
from tests.croppers import CropTest

TESTS = [
    [
        {"keys": ["img"], "roi_center": [1, 1], "roi_size": [2, 2]},
        (1, 3, 3),
        (1, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_center": [1, 1, 1], "roi_size": [2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_start": [0, 0, 0], "roi_end": [2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_start": [0, 0], "roi_end": [2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 3),
        (slice(None), slice(None, 2), slice(None, 2), slice(None)),
    ],
    [
        {"keys": ["img"], "roi_start": [0, 0, 0, 0, 0], "roi_end": [2, 2, 2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
        (slice(None), slice(None, 2), slice(None, 2), slice(None, 2)),
    ],
    [
        {"keys": ["img"], "roi_slices": [slice(s, e) for s, e in zip([-1, -2, 0], [None, None, 2])]},
        (3, 3, 3, 3),
        (3, 1, 2, 2),
        (slice(None), slice(-1, None), slice(-2, None), slice(0, 2)),
    ],
]


class TestSpatialCropd(CropTest):
    Cropper = SpatialCropd

    @parameterized.expand(TESTS)
    def test_shape(self, input_param, input_shape, expected_shape, same_area):
        self.crop_test(input_param, input_shape, expected_shape, same_area)

    @parameterized.expand(TESTS)
    def test_pending_ops(self, input_param, input_shape, _expected_shape, _same_area):
        self.crop_test_pending_ops(input_param, input_shape)


class TestSpatialCropdStringKeys(unittest.TestCase):
    """Tests for SpatialCropd with string dictionary keys for ROI parameters."""

    def test_string_roi_start_end(self):
        """String keys for roi_start and roi_end should resolve from data dict."""
        img = MetaTensor(torch.rand(3, 10, 10, 10))
        data = {"img": img, "roi_start_key": [0, 0, 0], "roi_end_key": [5, 5, 5]}
        cropper = SpatialCropd(keys="img", roi_start="roi_start_key", roi_end="roi_end_key")
        result = cropper(data)
        self.assertEqual(result["img"].shape, (3, 5, 5, 5))

    def test_string_roi_center_size(self):
        """String keys for roi_center and roi_size should resolve from data dict."""
        img = MetaTensor(torch.rand(1, 20, 20, 20))
        data = {"img": img, "center_key": [10, 10, 10], "size_key": [6, 6, 6]}
        cropper = SpatialCropd(keys="img", roi_center="center_key", roi_size="size_key")
        result = cropper(data)
        self.assertEqual(result["img"].shape, (1, 6, 6, 6))

    def test_mixed_string_and_direct(self):
        """Mix of string key and direct value for ROI params."""
        img = MetaTensor(torch.rand(1, 20, 20, 20))
        data = {"img": img, "center_key": [10, 10, 10]}
        cropper = SpatialCropd(keys="img", roi_center="center_key", roi_size=[4, 4, 4])
        result = cropper(data)
        self.assertEqual(result["img"].shape, (1, 4, 4, 4))

    def test_string_key_with_tensor(self):
        """String key resolving to a tensor value (e.g., from ApplyTransformToPoints output)."""
        img = MetaTensor(torch.rand(1, 20, 20, 20))
        # Simulate output from ApplyTransformToPoints: shape (C, N, dims) = (1, 1, 3)
        data = {
            "img": img,
            "roi_start_key": torch.tensor([[[2, 3, 4]]]),  # shape (1, 1, 3)
            "roi_end_key": torch.tensor([[[8, 9, 10]]]),  # shape (1, 1, 3)
        }
        cropper = SpatialCropd(keys="img", roi_start="roi_start_key", roi_end="roi_end_key")
        result = cropper(data)
        self.assertEqual(result["img"].shape, (1, 6, 6, 6))

    def test_string_key_with_float_tensor(self):
        """Float tensor values should be rounded before conversion."""
        img = MetaTensor(torch.rand(1, 20, 20, 20))
        data = {
            "img": img,
            "roi_start_key": torch.tensor([[[1.7, 2.3, 0.5]]]),
            "roi_end_key": torch.tensor([[[8.1, 9.8, 7.4]]]),
        }
        cropper = SpatialCropd(keys="img", roi_start="roi_start_key", roi_end="roi_end_key")
        result = cropper(data)
        # 1.7->2, 2.3->2, 0.5->0 (banker's rounding); 8.1->8, 9.8->10, 7.4->7
        self.assertEqual(result["img"].shape, (1, 6, 8, 7))

    def test_string_key_not_found(self):
        """Missing string key should raise KeyError."""
        img = MetaTensor(torch.rand(1, 10, 10, 10))
        data = {"img": img}
        cropper = SpatialCropd(keys="img", roi_start="missing_key", roi_end=[5, 5, 5])
        with self.assertRaises(KeyError):
            cropper(data)

    def test_requires_current_data(self):
        """requires_current_data should be True when string keys are used."""
        cropper_str = SpatialCropd(keys="img", roi_start="start_key", roi_end="end_key")
        self.assertTrue(cropper_str.requires_current_data)

        cropper_direct = SpatialCropd(keys="img", roi_start=[0, 0], roi_end=[5, 5])
        self.assertFalse(cropper_direct.requires_current_data)

    def test_string_key_same_as_direct(self):
        """String-key path should produce same result as direct-value path."""
        img_data = torch.rand(3, 10, 10, 10)
        roi_start = [1, 2, 3]
        roi_end = [5, 7, 8]

        # Direct values
        data_direct = {"img": MetaTensor(img_data.clone())}
        crop_direct = SpatialCropd(keys="img", roi_start=roi_start, roi_end=roi_end)
        result_direct = crop_direct(data_direct)

        # String keys
        data_str = {"img": MetaTensor(img_data.clone()), "start": roi_start, "end": roi_end}
        crop_str = SpatialCropd(keys="img", roi_start="start", roi_end="end")
        result_str = crop_str(data_str)

        self.assertEqual(result_direct["img"].shape, result_str["img"].shape)
        self.assertTrue(torch.allclose(result_direct["img"], result_str["img"]))

    def test_inverse_with_string_keys(self):
        """Inverse should work correctly when string keys are used."""
        img = MetaTensor(torch.rand(1, 10, 10, 10))
        data = {"img": img, "start": [2, 2, 2], "end": [6, 6, 6]}
        cropper = SpatialCropd(keys="img", roi_start="start", roi_end="end")
        result = cropper(data)
        self.assertEqual(result["img"].shape, (1, 4, 4, 4))

        inverted = cropper.inverse(result)
        self.assertEqual(inverted["img"].shape, (1, 10, 10, 10))
        self.assertEqual(inverted["img"].applied_operations, [])

    def test_pipeline_world_to_image_crop(self):
        """Integration test: TransformPointsWorldToImaged -> SpatialCropd with string keys."""
        # Create image with a 2x scaling affine: world coords = 2 * voxel coords
        affine = torch.tensor([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]], dtype=torch.float64)
        img = MetaTensor(torch.rand(1, 32, 32, 32), affine=affine)

        # World-space ROI: [4, 6, 8] to [20, 24, 28] -> voxel-space: [2, 3, 4] to [10, 12, 14]
        data = {
            "image": img,
            "roi_start": torch.tensor([[[4.0, 6.0, 8.0]]]),  # shape (1, 1, 3) world coords
            "roi_end": torch.tensor([[[20.0, 24.0, 28.0]]]),  # shape (1, 1, 3) world coords
        }

        # Step 1: convert world -> image coordinates
        w2i_start = TransformPointsWorldToImaged(keys="roi_start", refer_keys="image", dtype=torch.float64)
        w2i_end = TransformPointsWorldToImaged(keys="roi_end", refer_keys="image", dtype=torch.float64)
        data = w2i_start(data)
        data = w2i_end(data)

        # Step 2: crop using string keys
        cropper = SpatialCropd(keys="image", roi_start="roi_start", roi_end="roi_end")
        result = cropper(data)

        # Verify: voxel ROI should be [2,3,4] to [10,12,14] -> size [8, 9, 10]
        self.assertEqual(result["image"].shape, (1, 8, 9, 10))

    def test_multiple_image_keys(self):
        """String-key ROI should crop multiple images consistently."""
        data = {
            "img1": MetaTensor(torch.rand(1, 20, 20, 20)),
            "img2": MetaTensor(torch.rand(3, 20, 20, 20)),
            "start": [2, 2, 2],
            "end": [8, 8, 8],
        }
        cropper = SpatialCropd(keys=["img1", "img2"], roi_start="start", roi_end="end")
        result = cropper(data)
        self.assertEqual(result["img1"].shape, (1, 6, 6, 6))
        self.assertEqual(result["img2"].shape, (3, 6, 6, 6))

        # Inverse should restore both
        inverted = cropper.inverse(result)
        self.assertEqual(inverted["img1"].shape, (1, 20, 20, 20))
        self.assertEqual(inverted["img2"].shape, (3, 20, 20, 20))


if __name__ == "__main__":
    unittest.main()
