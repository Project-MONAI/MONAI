import os
import unittest
from typing import TypedDict
from unittest import skipUnless
from urllib import request

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.data.image_reader import WSIReader
from monai.utils import optional_import

_, has_osl = optional_import("openslide")


class SampleImage(TypedDict):
    name: str
    url: str
    height: int
    width: int


FILE_INFO: SampleImage = {
    "name": "CMU-1.tiff",
    "url": "http://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff",
    "height": 32914,
    "width": 46000,
}

TEST_CASE_0 = [FILE_INFO, (3, FILE_INFO["height"], FILE_INFO["width"])]

TEST_CASE_1 = [
    FILE_INFO,
    {"location": (FILE_INFO["height"] // 2, FILE_INFO["width"] // 2), "size": (2, 1), "level": 4},
    np.array([[[246], [246]], [[246], [246]], [[244], [244]]]),
]

TEST_CASE_2 = [
    FILE_INFO,
    {"location": (FILE_INFO["height"] // 2, FILE_INFO["width"] // 2), "size": (2, 1), "level": 2},
    np.array([[[246], [246]], [[246], [246]], [[246], [246]]]),
]

TEST_CASE_3 = [
    FILE_INFO,
    {
        "location": (FILE_INFO["height"] // 2, FILE_INFO["width"] // 2),
        "size": (8, 8),
        "level": 2,
        "grid_shape": (2, 1),
        "patch_size": 2,
    },
    np.array(
        [
            [[[246, 246], [246, 246]], [[246, 246], [246, 246]], [[246, 246], [246, 246]]],
            [[[246, 246], [246, 246]], [[246, 246], [246, 246]], [[246, 246], [246, 246]]],
        ]
    ),
]

TEST_CASE_4 = [
    FILE_INFO,
    {
        "location": (FILE_INFO["height"] // 2, FILE_INFO["width"] // 2),
        "size": (8, 8),
        "level": 2,
        "grid_shape": (2, 1),
        "patch_size": 1,
    },
    np.array([[[[246]], [[246]], [[246]]], [[[246]], [[246]], [[246]]]]),
]


class TestOpenSlideReader(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0])
    @skipUnless(has_osl, "Requires OpenSlide")
    def test_read_whole_image(self, file_info, expected_shape):
        filename = self.camelyon_data_download(file_info)
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj)
        self.assertTupleEqual(img.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @skipUnless(has_osl, "Requires OpenSlide")
    def test_read_region(self, file_info, patch_info, expected_img):
        filename = self.camelyon_data_download(file_info)
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj, **patch_info)
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    @parameterized.expand([TEST_CASE_3, TEST_CASE_4])
    @skipUnless(has_osl, "Requires OpenSlide")
    def test_read_patches(self, file_info, patch_info, expected_img):
        filename = self.camelyon_data_download(file_info)
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj, **patch_info)
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    def camelyon_data_download(self, file_info):
        from time import perf_counter

        filename = file_info["name"]
        if not os.path.exists(filename):
            print(f"Test image [{filename}] does not exist. Downloading...")
            t0 = perf_counter()
            request.urlretrieve(file_info["url"], filename)
            t1 = perf_counter()
            print(f"Elapsed time: {t1 - t0}s")
        return filename


if __name__ == "__main__":
    unittest.main()
