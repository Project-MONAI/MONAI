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

from monai.apps.pathology.data import SmartCachePatchWSIDataset
from monai.utils import optional_import
from tests.utils import download_url_or_skip_test, testing_data_config

_cucim, has_cim = optional_import("cucim")
has_cim = has_cim and hasattr(_cucim, "CuImage")

FILE_KEY = "wsi_img"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
base_name, extension = os.path.basename(f"{FILE_URL}"), ".tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + base_name + extension)

TEST_CASE_0 = [
    {
        "data": [
            {"image": FILE_PATH, "location": [0, 0], "label": [0]},
            {"image": FILE_PATH, "location": [0, 0], "label": [1]},
            {"image": FILE_PATH, "location": [0, 0], "label": [2]},
            {"image": FILE_PATH, "location": [0, 0], "label": [3]},
        ],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "transform": lambda x: x,
        "image_reader_name": "cuCIM",
        "replace_rate": 0.5,
        "cache_num": 2,
        "num_init_workers": 1,
        "num_replace_workers": 1,
        "copy_cache": False,
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[3]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[3]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0]]])},
    ],
]

TEST_CASE_1 = [
    {
        "data": [
            {"image": FILE_PATH, "location": [0, 0], "label": [[0, 0]]},
            {"image": FILE_PATH, "location": [0, 0], "label": [[1, 1]]},
            {"image": FILE_PATH, "location": [0, 0], "label": [[2, 2]]},
        ],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "transform": lambda x: x,
        "image_reader_name": "cuCIM",
        "replace_rate": 0.5,
        "cache_num": 2,
        "num_init_workers": 1,
        "num_replace_workers": 1,
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 0]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1, 1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1, 1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2, 2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2, 2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 0]]])},
    ],
]

TEST_CASE_2 = [
    {
        "data": [
            {"image": FILE_PATH, "location": [10004, 20004], "label": [0, 0, 0, 0]},
            {"image": FILE_PATH, "location": [10004, 20004], "label": [1, 1, 1, 1]},
            {"image": FILE_PATH, "location": [10004, 20004], "label": [2, 2, 2, 2]},
        ],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "transform": lambda x: x,
        "image_reader_name": "cuCIM",
        "replace_rate": 0.5,
        "cache_num": 2,
        "num_init_workers": 1,
        "num_replace_workers": 1,
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
    ],
]


class TestSmartCachePatchWSIDataset(unittest.TestCase):
    def setUp(self):
        hash_type = testing_data_config("images", FILE_KEY, "hash_type")
        hash_val = testing_data_config("images", FILE_KEY, "hash_val")
        download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    @skipUnless(has_cim, "Requires CuCIM")
    def test_read_patches(self, input_parameters, expected):
        dataset = SmartCachePatchWSIDataset(**input_parameters)
        self.assertEqual(len(dataset), input_parameters["cache_num"])
        total_num_samples = len(input_parameters["data"])
        num_epochs = int(
            np.ceil(total_num_samples / (input_parameters["cache_num"] * input_parameters["replace_rate"]))
        )

        dataset.start()
        i = 0
        for _ in range(num_epochs):
            for samples in dataset:
                n_patches = len(samples)
                self.assert_samples_expected(samples, expected[i : i + n_patches])
                i += n_patches
            dataset.update_cache()
        dataset.shutdown()

    def assert_samples_expected(self, samples, expected):
        for i, item in enumerate(samples):
            self.assertTupleEqual(item["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(item["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(item["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(item["image"], expected[i]["image"]))


if __name__ == "__main__":
    unittest.main()
