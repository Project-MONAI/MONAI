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

from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
from monai.data import MetaTensor


class TestExtractDataKeyFromMetaKeyd(unittest.TestCase):
    """Tests for ExtractDataKeyFromMetaKeyd covering both dict-based and MetaTensor-based metadata."""

    def test_extract_from_dict(self):
        """Test extracting keys from a plain metadata dictionary (image_only=False scenario)."""
        data = {
            "image": torch.zeros(1, 2, 2),
            "image_meta_dict": {"filename_or_obj": "image.nii", "spatial_shape": [2, 2]},
        }
        transform = ExtractDataKeyFromMetaKeyd(keys="filename_or_obj", meta_key="image_meta_dict")
        result = transform(data)
        self.assertIn("filename_or_obj", result)
        self.assertEqual(result["filename_or_obj"], "image.nii")
        self.assertEqual(result["image_meta_dict"]["filename_or_obj"], result["filename_or_obj"])

    def test_extract_from_metatensor(self):
        """Test extracting keys from a MetaTensor's .meta attribute (image_only=True scenario)."""
        mt = MetaTensor(torch.zeros(1, 2, 2))
        mt.meta["filename_or_obj"] = "image.nii"
        mt.meta["spatial_shape"] = [2, 2]
        data = {"image": mt}
        transform = ExtractDataKeyFromMetaKeyd(keys="filename_or_obj", meta_key="image")
        result = transform(data)
        self.assertIn("filename_or_obj", result)
        self.assertEqual(result["filename_or_obj"], "image.nii")
        self.assertEqual(result["image"].meta["filename_or_obj"], result["filename_or_obj"])

    def test_extract_multiple_keys_from_metatensor(self):
        """Test extracting multiple keys from a MetaTensor."""
        mt = MetaTensor(torch.zeros(1, 2, 2))
        mt.meta["filename_or_obj"] = "image.nii"
        mt.meta["spatial_shape"] = [2, 2]
        data = {"image": mt}
        transform = ExtractDataKeyFromMetaKeyd(keys=["filename_or_obj", "spatial_shape"], meta_key="image")
        result = transform(data)
        self.assertIn("filename_or_obj", result)
        self.assertIn("spatial_shape", result)
        self.assertEqual(result["filename_or_obj"], "image.nii")
        self.assertEqual(result["spatial_shape"], [2, 2])

    def test_extract_multiple_keys_from_dict(self):
        """Test extracting multiple keys from a plain dictionary."""
        data = {
            "image": torch.zeros(1, 2, 2),
            "image_meta_dict": {"filename_or_obj": "image.nii", "spatial_shape": [2, 2]},
        }
        transform = ExtractDataKeyFromMetaKeyd(keys=["filename_or_obj", "spatial_shape"], meta_key="image_meta_dict")
        result = transform(data)
        self.assertIn("filename_or_obj", result)
        self.assertIn("spatial_shape", result)
        self.assertEqual(result["filename_or_obj"], "image.nii")
        self.assertEqual(result["spatial_shape"], [2, 2])

    def test_missing_key_raises(self):
        """Test that a missing key raises KeyError when allow_missing_keys=False."""
        mt = MetaTensor(torch.zeros(1, 2, 2))
        mt.meta["filename_or_obj"] = "image.nii"
        data = {"image": mt}
        transform = ExtractDataKeyFromMetaKeyd(keys="nonexistent_key", meta_key="image")
        with self.assertRaises(KeyError):
            transform(data)

    def test_missing_key_allowed_metatensor(self):
        """Test that a missing key is silently skipped when allow_missing_keys=True with MetaTensor."""
        mt = MetaTensor(torch.zeros(1, 2, 2))
        mt.meta["filename_or_obj"] = "image.nii"
        data = {"image": mt}
        transform = ExtractDataKeyFromMetaKeyd(keys="nonexistent_key", meta_key="image", allow_missing_keys=True)
        result = transform(data)
        self.assertNotIn("nonexistent_key", result)

    def test_missing_key_allowed_dict(self):
        """Test that a missing key is silently skipped when allow_missing_keys=True with dict."""
        data = {"image": torch.zeros(1, 2, 2), "image_meta_dict": {"filename_or_obj": "image.nii"}}
        transform = ExtractDataKeyFromMetaKeyd(
            keys="nonexistent_key", meta_key="image_meta_dict", allow_missing_keys=True
        )
        result = transform(data)
        self.assertNotIn("nonexistent_key", result)

    def test_original_data_preserved_metatensor(self):
        """Test that the original MetaTensor remains in the data dictionary."""
        mt = MetaTensor(torch.ones(1, 2, 2))
        mt.meta["filename_or_obj"] = "image.nii"
        data = {"image": mt}
        transform = ExtractDataKeyFromMetaKeyd(keys="filename_or_obj", meta_key="image")
        result = transform(data)
        self.assertIn("image", result)
        self.assertIsInstance(result["image"], MetaTensor)
        self.assertTrue(torch.equal(result["image"], mt))


if __name__ == "__main__":
    unittest.main()
