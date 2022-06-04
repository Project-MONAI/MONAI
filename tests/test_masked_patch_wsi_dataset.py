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

import os
import unittest
from unittest import skipUnless

from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.data import MaskedPatchWSIDataset
from monai.utils import WSIPatchKeys, optional_import, set_determinism
from tests.utils import download_url_or_skip_test, testing_data_config

set_determinism(0)

cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
_, has_osl = optional_import("openslide")
_, has_tiff = optional_import("tifffile", name="imwrite")
_, has_codec = optional_import("imagecodecs")
has_tiff = has_tiff and has_codec


FILE_KEY = "wsi_img"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
base_name, extension = os.path.basename(f"{FILE_URL}"), ".tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + base_name + extension)

TEST_CASE_0 = [
    {"data": [{"image": FILE_PATH, WSIPatchKeys.LEVEL: 8, WSIPatchKeys.SIZE: (2, 2)}], "mask_level": 8},
    {
        "num_patches": 4256,
        "wsi_size": [32914, 46000],
        "mask_level": 8,
        "patch_level": 8,
        "mask_size": (128, 179),
        "patch_size": (2, 2),
    },
]


@skipUnless(has_cucim or has_osl or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():  # noqa: N802
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class MaskedPatchWSIDatasetTests:
    class Tests(unittest.TestCase):
        backend = None

        @parameterized.expand([TEST_CASE_0])
        def test_gen_patches(self, input_parameters, expected):
            dataset = MaskedPatchWSIDataset(reader=self.backend, **input_parameters)
            self.assertEqual(len(dataset), expected["num_patches"])
            for i, sample in enumerate(dataset):
                self.assertEqual(sample["metadata"][WSIPatchKeys.LEVEL], expected["patch_level"])
                assert_array_equal(sample["metadata"][WSIPatchKeys.SIZE], expected["patch_size"])
                assert_array_equal(sample["image"].shape[1:], expected["patch_size"])
                self.assertTrue(sample["metadata"][WSIPatchKeys.LOCATION][0] >= 0)
                self.assertTrue(sample["metadata"][WSIPatchKeys.LOCATION][0] < expected["wsi_size"][0])
                self.assertTrue(sample["metadata"][WSIPatchKeys.LOCATION][1] >= 0)
                self.assertTrue(sample["metadata"][WSIPatchKeys.LOCATION][1] < expected["wsi_size"][1])
                if i > 10:
                    break


@skipUnless(has_cucim, "Requires cucim")
class TestSlidingPatchWSIDatasetCuCIM(MaskedPatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class TestSlidingPatchWSIDatasetOpenSlide(MaskedPatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


if __name__ == "__main__":
    unittest.main()
