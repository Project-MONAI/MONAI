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

from monai.apps.pathology.data import MaskedInferenceWSIDataset
from monai.utils import optional_import
from tests.utils import download_url_or_skip_test, skip_if_quick, testing_data_config

_, has_cim = optional_import("cucim", name="CuImage")
_, has_osl = optional_import("openslide")

FILE_KEY = "wsi_img"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
base_name, extension = os.path.basename(f"{FILE_URL}"), ".tiff"
FILE_NAME = f"temp_{base_name}"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", FILE_NAME + extension)

MASK1 = os.path.join(os.path.dirname(__file__), "testing_data", "temp_tissue_mask1.npy")
MASK2 = os.path.join(os.path.dirname(__file__), "testing_data", "temp_tissue_mask2.npy")
MASK4 = os.path.join(os.path.dirname(__file__), "testing_data", "temp_tissue_mask4.npy")

HEIGHT = 32914
WIDTH = 46000


def prepare_data():

    mask = np.zeros((HEIGHT // 2, WIDTH // 2))
    mask[100, 100] = 1
    np.save(MASK1, mask)
    mask[100, 101] = 1
    np.save(MASK2, mask)
    mask[100:102, 100:102] = 1
    np.save(MASK4, mask)


TEST_CASE_0 = [
    {"data": [{"image": FILE_PATH, "mask": MASK1}], "patch_size": 1, "image_reader_name": "cuCIM"},
    [{"image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8), "name": FILE_NAME, "mask_location": [100, 100]}],
]

TEST_CASE_1 = [
    {"data": [{"image": FILE_PATH, "mask": MASK2}], "patch_size": 1, "image_reader_name": "cuCIM"},
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 101],
        },
    ],
]

TEST_CASE_2 = [
    {"data": [{"image": FILE_PATH, "mask": MASK4}], "patch_size": 1, "image_reader_name": "cuCIM"},
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 101],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [101, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [101, 101],
        },
    ],
]

TEST_CASE_3 = [
    {"data": [{"image": FILE_PATH, "mask": MASK1}], "patch_size": 2, "image_reader_name": "cuCIM"},
    [
        {
            "image": np.array(
                [[[243, 243], [243, 243]], [[243, 243], [243, 243]], [[243, 243], [243, 243]]], dtype=np.uint8
            ),
            "name": FILE_NAME,
            "mask_location": [100, 100],
        }
    ],
]

TEST_CASE_4 = [
    {
        "data": [{"image": FILE_PATH, "mask": MASK1}, {"image": FILE_PATH, "mask": MASK2}],
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 101],
        },
    ],
]


TEST_CASE_OPENSLIDE_0 = [
    {"data": [{"image": FILE_PATH, "mask": MASK1}], "patch_size": 1, "image_reader_name": "OpenSlide"},
    [{"image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8), "name": FILE_NAME, "mask_location": [100, 100]}],
]

TEST_CASE_OPENSLIDE_1 = [
    {"data": [{"image": FILE_PATH, "mask": MASK2}], "patch_size": 1, "image_reader_name": "OpenSlide"},
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": FILE_NAME,
            "mask_location": [100, 101],
        },
    ],
]


class TestMaskedInferenceWSIDataset(unittest.TestCase):
    def setUp(self):
        prepare_data()
        hash_type = testing_data_config("images", FILE_KEY, "hash_type")
        hash_val = testing_data_config("images", FILE_KEY, "hash_val")
        download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    @skipUnless(has_cim, "Requires CuCIM")
    @skip_if_quick
    def test_read_patches_cucim(self, input_parameters, expected):
        dataset = MaskedInferenceWSIDataset(**input_parameters)
        self.compare_samples_expected(dataset, expected)

    @parameterized.expand([TEST_CASE_OPENSLIDE_0, TEST_CASE_OPENSLIDE_1])
    @skipUnless(has_osl, "Requires OpenSlide")
    @skip_if_quick
    def test_read_patches_openslide(self, input_parameters, expected):
        dataset = MaskedInferenceWSIDataset(**input_parameters)
        self.compare_samples_expected(dataset, expected)

    def compare_samples_expected(self, dataset, expected):
        for i in range(len(dataset)):
            self.assertTupleEqual(dataset[i][0]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(dataset[i][0]["image"], expected[i]["image"]))
            self.assertEqual(dataset[i][0]["name"], expected[i]["name"])
            self.assertListEqual(dataset[i][0]["mask_location"], expected[i]["mask_location"])


if __name__ == "__main__":
    unittest.main()
