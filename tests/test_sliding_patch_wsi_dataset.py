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
from parameterized import parameterized

from monai.data import DataLoader, SlidingPatchWSIDataset
from monai.utils import first, optional_import, set_determinism
from tests.utils import download_url_or_skip_test, testing_data_config

set_determinism(0)

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

FILE_PATH_SMALL_0 = os.path.join(os.path.dirname(__file__), "testing_data", "temp_wsi_inference_0.tiff")
FILE_PATH_SMALL_1 = os.path.join(os.path.dirname(__file__), "testing_data", "temp_wsi_inference_1.tiff")
ARRAY_SMALL_0 = np.random.randint(low=0, high=255, size=(3, 4, 4), dtype=np.uint8)
ARRAY_SMALL_1 = np.random.randint(low=0, high=255, size=(3, 5, 5), dtype=np.uint8)

TEST_CASE_SMALL_0 = [
    {"data": [{"image": FILE_PATH_SMALL_0, "level": 0}], "size": (2, 2)},
    [
        {"image": ARRAY_SMALL_0[:, :2, :2]},
        {"image": ARRAY_SMALL_0[:, :2, 2:]},
        {"image": ARRAY_SMALL_0[:, 2:, :2]},
        {"image": ARRAY_SMALL_0[:, 2:, 2:]},
    ],
]

TEST_CASE_SMALL_1 = [
    {"data": [{"image": FILE_PATH_SMALL_0, "level": 0, "size": (2, 2)}]},
    [
        {"image": ARRAY_SMALL_0[:, :2, :2]},
        {"image": ARRAY_SMALL_0[:, :2, 2:]},
        {"image": ARRAY_SMALL_0[:, 2:, :2]},
        {"image": ARRAY_SMALL_0[:, 2:, 2:]},
    ],
]

TEST_CASE_SMALL_2 = [
    {"data": [{"image": FILE_PATH_SMALL_0, "level": 0}], "size": (2, 2), "overlap": 0.5},
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 1:3]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 1:3, 0:2]},
        {"image": ARRAY_SMALL_0[:, 1:3, 1:3]},
        {"image": ARRAY_SMALL_0[:, 1:3, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 1:3]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
    ],
]

TEST_CASE_SMALL_3 = [
    {"data": [{"image": FILE_PATH_SMALL_0, "level": 0}], "size": (3, 3), "overlap": 2.0 / 3.0},
    [
        {"image": ARRAY_SMALL_0[:, :3, :3]},
        {"image": ARRAY_SMALL_0[:, :3, 1:]},
        {"image": ARRAY_SMALL_0[:, 1:, :3]},
        {"image": ARRAY_SMALL_0[:, 1:, 1:]},
    ],
]

TEST_CASE_SMALL_4 = [
    {"data": [{"image": FILE_PATH_SMALL_0, "level": 0}, {"image": FILE_PATH_SMALL_1, "level": 0}], "size": (2, 2)},
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
        {"image": ARRAY_SMALL_1[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_1[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_1[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_1[:, 2:4, 2:4]},
    ],
]

TEST_CASE_SMALL_5 = [
    {
        "data": [
            {"image": FILE_PATH_SMALL_0, "level": 0, "size": (2, 2)},
            {"image": FILE_PATH_SMALL_1, "level": 0, "size": (3, 3)},
        ]
    },
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
        {"image": ARRAY_SMALL_1[:, 0:3, 0:3]},
    ],
]

TEST_CASE_SMALL_6 = [
    {
        "data": [
            {"image": FILE_PATH_SMALL_0, "level": 1, "size": (1, 1)},
            {"image": FILE_PATH_SMALL_1, "level": 2, "size": (4, 4)},
        ],
        "size": (2, 2),
        "level": 0,
    },
    [
        {"image": ARRAY_SMALL_0[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_0[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_0[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_0[:, 2:4, 2:4]},
        {"image": ARRAY_SMALL_1[:, 0:2, 0:2]},
        {"image": ARRAY_SMALL_1[:, 0:2, 2:4]},
        {"image": ARRAY_SMALL_1[:, 2:4, 0:2]},
        {"image": ARRAY_SMALL_1[:, 2:4, 2:4]},
    ],
]


TEST_CASE_SMALL_7 = [
    {"data": [{"image": FILE_PATH_SMALL_0, "level": 0, "size": (2, 2)}], "offset": (1, 0)},
    [{"image": ARRAY_SMALL_0[:, 1:3, :2]}, {"image": ARRAY_SMALL_0[:, 1:3, 2:]}],
]

TEST_CASE_SMALL_8 = [
    {"data": [{"image": FILE_PATH_SMALL_0, "level": 0, "size": (2, 2)}], "offset": "random", "offset_limits": (0, 2)},
    [{"image": ARRAY_SMALL_0[:, 1:3, :2]}, {"image": ARRAY_SMALL_0[:, 1:3, 2:]}],
]

TEST_CASE_SMALL_9 = [
    {
        "data": [{"image": FILE_PATH_SMALL_0, "level": 0, "size": (2, 2)}],
        "offset": "random",
        "offset_limits": ((0, 3), (0, 2)),
    },
    [{"image": ARRAY_SMALL_0[:, :2, 1:3]}, {"image": ARRAY_SMALL_0[:, 2:, 1:3]}],
]

TEST_CASE_LARGE_0 = [
    {"data": [{"image": FILE_PATH, "level": 8, "size": (64, 50)}]},
    [
        {"step_loc": (0, 0), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 1), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 2), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 0), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 1), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 2), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
    ],
]

TEST_CASE_LARGE_1 = [
    {
        "data": [
            {"image": FILE_PATH, "level": 8, "size": (64, 50)},
            {"image": FILE_PATH, "level": 7, "size": (125, 110)},
        ]
    },
    [
        {"step_loc": (0, 0), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 1), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 2), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 0), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 1), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (1, 2), "size": (64, 50), "level": 8, "ratio": 257.06195068359375},
        {"step_loc": (0, 0), "size": (125, 110), "level": 7, "ratio": 128.10186767578125},
        {"step_loc": (0, 1), "size": (125, 110), "level": 7, "ratio": 128.10186767578125},
        {"step_loc": (0, 2), "size": (125, 110), "level": 7, "ratio": 128.10186767578125},
        {"step_loc": (1, 0), "size": (125, 110), "level": 7, "ratio": 128.10186767578125},
        {"step_loc": (1, 1), "size": (125, 110), "level": 7, "ratio": 128.10186767578125},
        {"step_loc": (1, 2), "size": (125, 110), "level": 7, "ratio": 128.10186767578125},
    ],
]


@skipUnless(has_cucim or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():  # noqa: N802
    for info in [(ARRAY_SMALL_0, FILE_PATH_SMALL_0), (ARRAY_SMALL_1, FILE_PATH_SMALL_1)]:
        array = info[0].transpose([1, 2, 0])
        imwrite(info[1], array, shape=array.shape, photometric="rgb")
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class SlidingPatchWSIDatasetTests:
    class Tests(unittest.TestCase):
        backend = None

        @parameterized.expand(
            [
                TEST_CASE_SMALL_0,
                TEST_CASE_SMALL_1,
                TEST_CASE_SMALL_2,
                TEST_CASE_SMALL_3,
                TEST_CASE_SMALL_4,
                TEST_CASE_SMALL_5,
                TEST_CASE_SMALL_6,
                TEST_CASE_SMALL_7,
                TEST_CASE_SMALL_8,
                TEST_CASE_SMALL_9,
            ]
        )
        def test_read_patches(self, input_parameters, expected):
            if self.backend == "openslide":
                return
            dataset = SlidingPatchWSIDataset(reader=self.backend, **input_parameters)
            self.assertEqual(len(dataset), len(expected))
            for i, sample in enumerate(dataset):
                self.assertTupleEqual(sample["image"].shape, expected[i]["image"].shape)

        @parameterized.expand([TEST_CASE_LARGE_0, TEST_CASE_LARGE_1])
        def test_read_patches_large(self, input_parameters, expected):
            dataset = SlidingPatchWSIDataset(reader=self.backend, **input_parameters)
            self.assertEqual(len(dataset), len(expected))
            for i, sample in enumerate(dataset):
                self.assertEqual(sample["metadata"]["patch_level"], expected[i]["level"])
                self.assertTupleEqual(tuple(sample["metadata"]["patch_size"]), expected[i]["size"])
                steps = [round(expected[i]["ratio"] * s) for s in expected[i]["size"]]
                expected_location = tuple(expected[i]["step_loc"][j] * steps[j] for j in range(len(steps)))
                self.assertTupleEqual(tuple(sample["metadata"]["patch_location"]), expected_location)


@skipUnless(has_cucim, "Requires cucim")
class TestSlidingPatchWSIDatasetCuCIM(SlidingPatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class TestSlidingPatchWSIDatasetOpenSlide(SlidingPatchWSIDatasetTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


if __name__ == "__main__":
    unittest.main()
