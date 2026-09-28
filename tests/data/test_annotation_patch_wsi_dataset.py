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

import os
import tempfile
import unittest
from pathlib import Path
from unittest import skipUnless

import numpy as np

from monai.data import AnnotationPatchWSIDataset
from monai.utils import CommonKeys, optional_import, set_determinism
from tests.test_utils import download_url_or_skip_test, testing_data_config

set_determinism(0)

cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
_, has_osl = optional_import("openslide")
_, has_tiff = optional_import("tifffile", name="imwrite")
_, has_codec = optional_import("imagecodecs")
has_tiff = has_tiff and has_codec

FILE_KEY = "wsi_generic_tiff"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
TESTS_PATH = Path(__file__).parents[1]
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", f"temp_{FILE_KEY}.tiff")


@skipUnless(has_cucim or has_osl or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class AnnotationPatchWSIDatasetTests:
    class Tests(unittest.TestCase):
        backend = None

        def setUp(self):
            # Create a temporary annotation mask with 3 classes
            self.mask = np.zeros((128, 179), dtype=np.int32)
            self.mask[10:50, 10:80] = 1  # class 1: tumor
            self.mask[60:100, 90:170] = 2  # class 2: stroma
            self.mask[100:120, 20:60] = 3  # class 3: necrosis

            self.mask_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            np.save(self.mask_file.name, self.mask)

        def tearDown(self):
            os.unlink(self.mask_file.name)

        def test_uniform_sampling(self):
            """Test that patches are sampled from all classes with uniform weights."""
            data = [{"image": FILE_PATH, "mask": self.mask_file.name}]
            dataset = AnnotationPatchWSIDataset(
                data=data,
                patch_size=(2, 2),
                patch_level=8,
                mask_level=8,
                num_patches_per_image=50,
                reader=self.backend,
                seed=42,
            )
            self.assertEqual(len(dataset), 50)

            # Check that labels are from the expected classes
            labels = set()
            for i in range(len(dataset)):
                sample = dataset[i]
                self.assertIn(CommonKeys.IMAGE, sample)
                self.assertIn(CommonKeys.LABEL, sample)
                labels.add(int(sample[CommonKeys.LABEL].item()))
            # With 50 samples and uniform weights, we should see all 3 classes
            self.assertEqual(labels, {1, 2, 3})

        def test_weighted_sampling(self):
            """Test that sampling weights control class distribution."""
            data = [{"image": FILE_PATH, "mask": self.mask_file.name}]
            # Only sample from class 1
            dataset = AnnotationPatchWSIDataset(
                data=data,
                patch_size=(2, 2),
                patch_level=8,
                mask_level=8,
                num_patches_per_image=20,
                sampling_weights={1: 1.0, 2: 0.0, 3: 0.0},
                reader=self.backend,
                seed=42,
            )
            self.assertEqual(len(dataset), 20)
            for i in range(len(dataset)):
                sample = dataset[i]
                self.assertEqual(int(sample[CommonKeys.LABEL].item()), 1)

        def test_mask_as_array(self):
            """Test that mask can be passed directly as a numpy array."""
            data = [{"image": FILE_PATH, "mask": self.mask}]
            dataset = AnnotationPatchWSIDataset(
                data=data,
                patch_size=(2, 2),
                patch_level=8,
                mask_level=8,
                num_patches_per_image=10,
                reader=self.backend,
                seed=42,
            )
            self.assertEqual(len(dataset), 10)

        def test_empty_mask(self):
            """Test that an all-zero mask produces no patches."""
            empty_mask = np.zeros((128, 179), dtype=np.int32)
            data = [{"image": FILE_PATH, "mask": empty_mask}]
            dataset = AnnotationPatchWSIDataset(
                data=data,
                patch_size=(2, 2),
                patch_level=8,
                mask_level=8,
                num_patches_per_image=10,
                reader=self.backend,
                seed=42,
            )
            self.assertEqual(len(dataset), 0)

        def test_metadata_keys(self):
            """Test that output contains expected metadata."""
            data = [{"image": FILE_PATH, "mask": self.mask_file.name}]
            dataset = AnnotationPatchWSIDataset(
                data=data,
                patch_size=(2, 2),
                patch_level=8,
                mask_level=8,
                num_patches_per_image=5,
                reader=self.backend,
                seed=42,
            )
            sample = dataset[0]
            self.assertIn(CommonKeys.IMAGE, sample)
            self.assertIn(CommonKeys.LABEL, sample)


@skipUnless(has_cucim, "Requires cucim")
class AnnotationPatchWSIDatasetCuCIMTests(AnnotationPatchWSIDatasetTests.Tests):
    backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class AnnotationPatchWSIDatasetOpenSlideTests(AnnotationPatchWSIDatasetTests.Tests):
    backend = "openslide"


@skipUnless(has_tiff, "Requires tifffile")
class AnnotationPatchWSIDatasetTiffFileTests(AnnotationPatchWSIDatasetTests.Tests):
    backend = "tifffile"


if __name__ == "__main__":
    unittest.main()
