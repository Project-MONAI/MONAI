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

from monai.apps.pathology.data import PatchWSIDataset as PatchWSIDatasetDeprecated
from monai.data import PatchWSIDataset
from monai.data.wsi_reader import CuCIMWSIReader, OpenSlideWSIReader
from monai.utils import deprecated, optional_import
from tests.utils import download_url_or_skip_test, testing_data_config

cucim, has_cim = optional_import("cucim")
has_cim = has_cim and hasattr(cucim, "CuImage")
openslide, has_osl = optional_import("openslide")
imwrite, has_tiff = optional_import("tifffile", name="imwrite")
_, has_codec = optional_import("imagecodecs")
has_tiff = has_tiff and has_codec

FILE_KEY = "wsi_img"
FILE_URL = testing_data_config("images", FILE_KEY, "url")
base_name, extension = os.path.basename(f"{FILE_URL}"), ".tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + base_name + extension)

TEST_CASE_DEP_0 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_DEP_0_L1 = [
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

TEST_CASE_DEP_0_L2 = [
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


TEST_CASE_DEP_1 = [
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


TEST_CASE_DEP_1_L0 = [
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


TEST_CASE_DEP_1_L1 = [
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
TEST_CASE_DEP_2 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": 1,
        "grid_shape": 1,
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_DEP_3 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [[[0, 1], [1, 0]]]}],
        "region_size": 1,
        "grid_shape": 1,
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])}],
]

TEST_CASE_DEP_OPENSLIDE_0 = [
    {
        "data": [{"image": FILE_PATH, "location": [0, 0], "label": [1]}],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "image_reader_name": "OpenSlide",
    },
    [{"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])}],
]

TEST_CASE_DEP_OPENSLIDE_0_L0 = [
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

TEST_CASE_DEP_OPENSLIDE_0_L1 = [
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


TEST_CASE_DEP_OPENSLIDE_0_L2 = [
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

TEST_CASE_DEP_OPENSLIDE_1 = [
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


TEST_CASE_0 = [
    {"data": [{"image": FILE_PATH, "patch_location": [0, 0], "label": [1], "patch_level": 0}], "patch_size": (1, 1)},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_0_L1 = [
    {"data": [{"image": FILE_PATH, "patch_location": [0, 0], "label": [1]}], "patch_size": (1, 1), "patch_level": 1},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_0_L2 = [
    {"data": [{"image": FILE_PATH, "patch_location": [0, 0], "label": [1]}], "patch_size": (1, 1), "patch_level": 1},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]
TEST_CASE_1 = [
    {"data": [{"image": FILE_PATH, "patch_location": [0, 0], "patch_size": 1, "label": [1]}]},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_2 = [
    {"data": [{"image": FILE_PATH, "patch_location": [0, 0], "label": [1]}], "patch_size": 1, "patch_level": 0},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([1])},
]

TEST_CASE_3 = [
    {"data": [{"image": FILE_PATH, "patch_location": [0, 0], "label": [[[0, 1], [1, 0]]]}], "patch_size": 1},
    {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])},
]

TEST_CASE_4 = [
    {
        "data": [
            {"image": FILE_PATH, "patch_location": [0, 0], "label": [[[0, 1], [1, 0]]]},
            {"image": FILE_PATH, "patch_location": [0, 0], "label": [[[1, 0], [0, 0]]]},
        ],
        "patch_size": 1,
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1, 0], [0, 0]]])},
    ],
]

TEST_CASE_5 = [
    {
        "data": [
            {
                "image": FILE_PATH,
                "patch_location": [0, 0],
                "label": [[[0, 1], [1, 0]]],
                "patch_size": 1,
                "patch_level": 1,
            },
            {
                "image": FILE_PATH,
                "patch_location": [100, 100],
                "label": [[[1, 0], [0, 0]]],
                "patch_size": 1,
                "patch_level": 1,
            },
        ]
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])},
        {"image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8), "label": np.array([[[1, 0], [0, 0]]])},
    ],
]


@skipUnless(has_cim or has_osl or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


@deprecated(since="0.8", msg_suffix="use tests for `monai.data.PatchWSIDataset` instead, `PatchWSIDatasetTests`.")
class TestPatchWSIDatasetDeprecated(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_DEP_0,
            TEST_CASE_DEP_0_L1,
            TEST_CASE_DEP_0_L2,
            TEST_CASE_DEP_1,
            TEST_CASE_DEP_1_L0,
            TEST_CASE_DEP_1_L1,
            TEST_CASE_DEP_2,
            TEST_CASE_DEP_3,
        ]
    )
    @skipUnless(has_cim, "Requires CuCIM")
    def test_read_patches_cucim(self, input_parameters, expected):
        dataset = PatchWSIDatasetDeprecated(**input_parameters)
        samples = dataset[0]
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))

    @parameterized.expand(
        [
            TEST_CASE_DEP_OPENSLIDE_0,
            TEST_CASE_DEP_OPENSLIDE_0_L0,
            TEST_CASE_DEP_OPENSLIDE_0_L1,
            TEST_CASE_DEP_OPENSLIDE_0_L2,
            TEST_CASE_DEP_OPENSLIDE_1,
        ]
    )
    @skipUnless(has_osl, "Requires OpenSlide")
    def test_read_patches_openslide(self, input_parameters, expected):
        dataset = PatchWSIDatasetDeprecated(**input_parameters)
        samples = dataset[0]
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))


class PatchWSIDatasetTests:
    class Tests(unittest.TestCase):
        backend = None

        @parameterized.expand([TEST_CASE_0, TEST_CASE_0_L1, TEST_CASE_0_L2, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
        def test_read_patches_str(self, input_parameters, expected):
            dataset = PatchWSIDataset(reader=self.backend, **input_parameters)
            sample = dataset[0]
            self.assertTupleEqual(sample["label"].shape, expected["label"].shape)
            self.assertTupleEqual(sample["image"].shape, expected["image"].shape)
            self.assertIsNone(assert_array_equal(sample["label"], expected["label"]))
            self.assertIsNone(assert_array_equal(sample["image"], expected["image"]))

        @parameterized.expand([TEST_CASE_0, TEST_CASE_0_L1, TEST_CASE_0_L2, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
        def test_read_patches_class(self, input_parameters, expected):
            if self.backend == "openslide":
                reader = OpenSlideWSIReader
            elif self.backend == "cucim":
                reader = CuCIMWSIReader
            else:
                raise ValueError("Unsupported backend: {self.backend}")
            dataset = PatchWSIDataset(reader=reader, **input_parameters)
            sample = dataset[0]
            self.assertTupleEqual(sample["label"].shape, expected["label"].shape)
            self.assertTupleEqual(sample["image"].shape, expected["image"].shape)
            self.assertIsNone(assert_array_equal(sample["label"], expected["label"]))
            self.assertIsNone(assert_array_equal(sample["image"], expected["image"]))

        @parameterized.expand([TEST_CASE_0, TEST_CASE_0_L1, TEST_CASE_0_L2, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
        def test_read_patches_object(self, input_parameters, expected):
            if self.backend == "openslide":
                reader = OpenSlideWSIReader(level=input_parameters.get("patch_level", 0))
            elif self.backend == "cucim":
                reader = CuCIMWSIReader(level=input_parameters.get("patch_level", 0))
            else:
                raise ValueError("Unsupported backend: {self.backend}")
            dataset = PatchWSIDataset(reader=reader, **input_parameters)
            sample = dataset[0]
            self.assertTupleEqual(sample["label"].shape, expected["label"].shape)
            self.assertTupleEqual(sample["image"].shape, expected["image"].shape)
            self.assertIsNone(assert_array_equal(sample["label"], expected["label"]))
            self.assertIsNone(assert_array_equal(sample["image"], expected["image"]))

        @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
        def test_read_patches_str_multi(self, input_parameters, expected):
            dataset = PatchWSIDataset(reader=self.backend, **input_parameters)
            for i in range(len(dataset)):
                self.assertTupleEqual(dataset[i]["label"].shape, expected[i]["label"].shape)
                self.assertTupleEqual(dataset[i]["image"].shape, expected[i]["image"].shape)
                self.assertIsNone(assert_array_equal(dataset[i]["label"], expected[i]["label"]))
                self.assertIsNone(assert_array_equal(dataset[i]["image"], expected[i]["image"]))


@skipUnless(has_cim, "Requires cucim")
class TestPatchWSIDatasetCuCIM(PatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class TestPatchWSIDatasetOpenSlide(PatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


if __name__ == "__main__":
    unittest.main()
