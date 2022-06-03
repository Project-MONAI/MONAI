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

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.apps.pathology.data import PatchWSIDataset
from monai.utils import optional_import
from tests.utils import download_url_or_skip_test, testing_data_config

_cucim, has_cim = optional_import("cucim")
has_cim = has_cim and hasattr(_cucim, "CuImage")
_, has_osl = optional_import("openslide")

FILE_KEY = "wsi_img"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
base_name, extension = os.path.basename(f"{FILE_URL}"), ".tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + base_name + extension)

TEST_CASE_0 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_0_L1 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "level": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_0_L2 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "level": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]


TEST_CASE_1 = [
    {
        "data": [{"image": FILE_PATH, "location": [10004, 20004], "label": [0, 0, 0, 1]}],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]


TEST_CASE_1_L0 = [
    {
        "data": [{"image": FILE_PATH, "location": [10004, 20004], "label": [0, 0, 0, 1]}],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "level": 0,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]


TEST_CASE_1_L1 = [
    {
        "data": [{"image": FILE_PATH, "location": [10004, 20004], "label": [0, 0, 0, 1]}],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "level": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[248]], [[246]], [[249]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[196]], [[187]], [[192]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[243]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[242]], [[243]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]
TEST_CASE_2 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": 1,
        "grid_shape": 1,
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_3 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [[[0, 1], [1, 0]]]}],
        "region_size": 1,
        "grid_shape": 1,
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])}],
]

TEST_CASE_OPENSLIDE_0 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "image_reader_name": "OpenSlide",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_OPENSLIDE_0_L0 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "level": 0,
        "image_reader_name": "OpenSlide",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_OPENSLIDE_0_L1 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "level": 1,
        "image_reader_name": "OpenSlide",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]


TEST_CASE_OPENSLIDE_0_L2 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "level": 2,
        "image_reader_name": "OpenSlide",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_OPENSLIDE_1 = [
    {
        "data": [{"image": FILE_PATH, "location": [10004, 20004], "label": [0, 0, 0, 1]}],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "image_reader_name": "OpenSlide",
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]


class TestPatchWSIDataset(unittest.TestCase):
    def setUp(self):
        hash_type = testing_data_config("images", FILE_KEY, "hash_type")
        hash_val = testing_data_config("images", FILE_KEY, "hash_val")
        download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)

    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_0_L1,
            TEST_CASE_0_L2,
            TEST_CASE_1,
            TEST_CASE_1_L0,
            TEST_CASE_1_L1,
            TEST_CASE_2,
            TEST_CASE_3,
        ]
    )
    @skipUnless(has_cim, "Requires CuCIM")
    def test_read_patches_cucim(self, input_parameters, expected):
        dataset = PatchWSIDataset(**input_parameters)
        samples = dataset[0]
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))

    @parameterized.expand(
        [
            TEST_CASE_OPENSLIDE_0,
            TEST_CASE_OPENSLIDE_0_L0,
            TEST_CASE_OPENSLIDE_0_L1,
            TEST_CASE_OPENSLIDE_0_L2,
            TEST_CASE_OPENSLIDE_1,
        ]
    )
    @skipUnless(has_osl, "Requires OpenSlide")
    def test_read_patches_openslide(self, input_parameters, expected):
        dataset = PatchWSIDataset(**input_parameters)
        samples = dataset[0]
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))


if __name__ == "__main__":
    unittest.main()
