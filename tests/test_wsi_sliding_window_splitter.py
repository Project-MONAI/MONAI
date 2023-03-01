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
import unittest
from unittest import skipUnless

from parameterized import parameterized
from torch.nn.functional import pad

from monai.inferers import WSISlidingWindowSplitter
from tests.utils import assert_allclose, download_url_or_skip_test, optional_import, testing_data_config


cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
openslide, has_osl = optional_import("openslide")
imwrite, has_tiff = optional_import("tifffile", name="imwrite")
_, has_codec = optional_import("imagecodecs")
has_tiff = has_tiff and has_codec

FILE_KEY = "wsi_img"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
base_name, extension = os.path.basename(f"{FILE_URL}"), ".tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + base_name + extension)

HEIGHT = 32914
WIDTH = 46000

# ----------------------------------------------------------------------------
# Primary use test cases
# ----------------------------------------------------------------------------
# no-overlapping 2x2
TEST_CASE_0 = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "reader": "OpenSlide"},
]


@skipUnless(has_cucim or has_osl or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class WSISlidingWindowSplitterTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0,
        ]
    )
    def test_split_patches(self, image, arguments):
        patches = WSISlidingWindowSplitter(**arguments)(image)
        # self.assertEqual(len(patches), len(expected))
        for i, sample in enumerate(patches):
            patch = sample[0]
            loc = sample[1]
            print(f"{type(patch)=}")
            print(f"{patch.shape=}")
            print(f"{patch.sum()=}")
            print(f"{type(loc)=}")
            if i == 10:
                break


if __name__ == "__main__":
    unittest.main()
