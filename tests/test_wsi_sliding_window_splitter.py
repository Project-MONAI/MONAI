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

import torch
from parameterized import parameterized

from monai.data import CuCIMWSIReader, ImageReader, OpenSlideWSIReader, WSIReader
from monai.inferers import WSISlidingWindowSplitter
from tests.utils import download_url_or_skip_test, optional_import, testing_data_config

cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
_, has_osl = optional_import("openslide")

WSI_READER_STR = None
WSI_READER_CLASS: type[CuCIMWSIReader] | type[OpenSlideWSIReader] | None = None
if has_cucim:
    WSI_READER_STR = "cuCIM"
    WSI_READER_CLASS = CuCIMWSIReader
elif has_osl:
    WSI_READER_STR = "OpenSlide"
    WSI_READER_CLASS = OpenSlideWSIReader


FILE_KEY = "wsi_img"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
base_name, extension = os.path.basename(f"{FILE_URL}"), ".tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + base_name + extension)

HEIGHT = 32914
WIDTH = 46000

# ----------------------------------------------------------------------------
# Primary use test cases
# ----------------------------------------------------------------------------
TEST_CASE_0_BASE = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]

TEST_CASE_1_BASE = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "patch_level": 1, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]

# Check readers
if WSI_READER_STR is not None:
    TEST_CASE_2_READER = [
        FILE_PATH,
        {"patch_size": (1000, 1000), "reader": WSIReader(backend=WSI_READER_STR)},
        {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
    ]
else:
    TEST_CASE_2_READER = []
TEST_CASE_3_READER = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "reader": WSIReader, "reader_kwargs": dict(backend=WSI_READER_STR)},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]
TEST_CASE_4_READER = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "reader": WSI_READER_CLASS},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]
TEST_CASE_5_READER = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "patch_level": 1, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]

# Check overlaps
TEST_CASE_6_OVERLAP = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.0, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1000), (0, 2000), (0, 3000)]},
]
TEST_CASE_7_OVERLAP = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.5, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 500), (0, 1000), (0, 1500)]},
]
TEST_CASE_8_OVERLAP = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "overlap": 0.999, "reader": WSI_READER_STR},
    {"locations": [(0, 0), (0, 1), (0, 2), (0, 3)]},
]

# Filtering functions test cases
def gen_location_filter(locations):
    def my_filter(patch, loc):
        if loc in locations:
            return False
        return True

    return my_filter


TEST_CASE_9_FILTER = [
    FILE_PATH,
    {"patch_size": (1000, 1000), "reader": WSI_READER_STR, "filter_fn": gen_location_filter([(0, 0), (0, 2000)])},
    {"locations": [(0, 1000), (0, 3000)]},
]


# ----------------------------------------------------------------------------
# Error test cases
# ----------------------------------------------------------------------------
def extra_parameter_filter(patch, location, extra):
    return


def missing_parameter_filter(patch):
    return


# invalid overlap: 1.0
TEST_CASE_ERROR_0 = [FILE_PATH, {"patch_size": (2, 2), "overlap": 1.0, "reader": WSI_READER_STR}, ValueError]
# invalid overlap: negative
TEST_CASE_ERROR_1 = [FILE_PATH, {"patch_size": (2, 2), "overlap": -0.1, "reader": WSI_READER_STR}, ValueError]

# invalid offset: positive and larger than image size
TEST_CASE_ERROR_2 = [
    FILE_PATH,
    {"patch_size": (2, 2), "offset": max(HEIGHT, WIDTH), "reader": WSI_READER_STR},
    ValueError,
]
# invalid offset: negative and larger than patch size (in magnitude)
TEST_CASE_ERROR_3 = [FILE_PATH, {"patch_size": (2, 2), "offset": -3, "reader": WSI_READER_STR}, ValueError]

# invalid reader
TEST_CASE_ERROR_4 = [FILE_PATH, {"patch_size": (2, 2), "offset": -3, "reader": ImageReader}, ValueError]

# invalid filter function: with more than two positional parameters
TEST_CASE_ERROR_5 = [
    FILE_PATH,
    {"patch_size": (2, 2), "filter_fn": extra_parameter_filter, "reader": WSI_READER_STR},
    ValueError,
]
# invalid filter function: with less than two positional parameters
TEST_CASE_ERROR_6 = [
    FILE_PATH,
    {"patch_size": (2, 2), "filter_fn": missing_parameter_filter, "reader": WSI_READER_STR},
    ValueError,
]
# invalid filter function: non-callable
TEST_CASE_ERROR_7 = [FILE_PATH, {"patch_size": (2, 2), "filter_fn": 1, "reader": WSI_READER_STR}, ValueError]


@skipUnless(WSI_READER_STR, "Requires cucim or openslide!")
def setUpModule():
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class WSISlidingWindowSplitterTests(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0_BASE,
            TEST_CASE_1_BASE,
            TEST_CASE_2_READER,
            TEST_CASE_3_READER,
            TEST_CASE_4_READER,
            TEST_CASE_5_READER,
            TEST_CASE_6_OVERLAP,
            TEST_CASE_7_OVERLAP,
            TEST_CASE_8_OVERLAP,
            TEST_CASE_9_FILTER,
        ]
    )
    def test_split_patches(self, image, arguments, expected):
        patches = WSISlidingWindowSplitter(**arguments)(image)
        for sample, expected_loc in zip(patches, expected["locations"]):
            patch = sample[0]
            loc = sample[1]
            self.assertTrue(isinstance(patch, torch.Tensor))
            self.assertTupleEqual(patch.shape[1:], arguments["patch_size"])
            self.assertTrue(isinstance(loc, tuple))
            self.assertTupleEqual(loc, expected_loc)

    @parameterized.expand(
        [
            TEST_CASE_ERROR_0,
            TEST_CASE_ERROR_1,
            TEST_CASE_ERROR_2,
            TEST_CASE_ERROR_3,
            TEST_CASE_ERROR_4,
            TEST_CASE_ERROR_5,
            TEST_CASE_ERROR_6,
            TEST_CASE_ERROR_7,
        ]
    )
    def test_split_patches_errors(self, image, arguments, expected_error):
        with self.assertRaises(expected_error):
            patches = WSISlidingWindowSplitter(**arguments)(image)
            patches = list(patches)


if __name__ == "__main__":
    unittest.main()
