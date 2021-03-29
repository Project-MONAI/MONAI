import os
import unittest
from unittest import skipUnless

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.apps.utils import download_url
from monai.data.image_reader import WSIReader
from monai.utils import optional_import

_, has_osl = optional_import("openslide")


FILE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", os.path.basename(FILE_URL))

HEIGHT = 32914
WIDTH = 46000

TEST_CASE_0 = [FILE_PATH, (3, HEIGHT, WIDTH)]

TEST_CASE_1 = [
    FILE_PATH,
    {"location": (HEIGHT // 2, WIDTH // 2), "size": (2, 1), "level": 0},
    np.array([[[246], [246]], [[246], [246]], [[246], [246]]]),
]

TEST_CASE_2 = [
    FILE_PATH,
    {"location": (0, 0), "size": (2, 1), "level": 2},
    np.array([[[239], [239]], [[239], [239]], [[239], [239]]]),
]

TEST_CASE_3 = [
    FILE_PATH,
    {
        "location": (0, 0),
        "size": (8, 8),
        "level": 2,
        "grid_shape": (2, 1),
        "patch_size": 2,
    },
    np.array(
        [
            [[[239, 239], [239, 239]], [[239, 239], [239, 239]], [[239, 239], [239, 239]]],
            [[[242, 242], [242, 243]], [[242, 242], [242, 243]], [[242, 242], [242, 243]]],
        ]
    ),
]

TEST_CASE_4 = [
    FILE_PATH,
    {
        "location": (0, 0),
        "size": (8, 8),
        "level": 2,
        "grid_shape": (2, 1),
        "patch_size": 1,
    },
    np.array([[[[239]], [[239]], [[239]]], [[[243]], [[243]], [[243]]]]),
]


class TestOpenSlideReader(unittest.TestCase):
    @skipUnless(has_osl, "Requires OpenSlide")
    def setUp(self):
        download_url(FILE_URL, FILE_PATH, '5a3cfd4fd725c50578ddb80b517b759f')

    @parameterized.expand([TEST_CASE_0])
    def test_read_whole_image(self, file_path, expected_shape):
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(file_path)
        img = reader.get_data(img_obj)[0]
        self.assertTupleEqual(img.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_read_region(self, file_path, patch_info, expected_img):
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(file_path)
        img = reader.get_data(img_obj, **patch_info)[0]
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    @parameterized.expand([TEST_CASE_3, TEST_CASE_4])
    def test_read_patches(self, file_path, patch_info, expected_img):
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(file_path)
        img = reader.get_data(img_obj, **patch_info)[0]
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))


if __name__ == "__main__":
    unittest.main()
