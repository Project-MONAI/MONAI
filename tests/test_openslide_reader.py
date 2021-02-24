import ftplib
import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.data.image_reader import WSIReader
from tests.utils import skip_if_quick

filename = "test_065.tif"

TEST_CASE_0 = [filename, (3, 53760, 77824)]

TEST_CASE_1 = [
    filename,
    {"location": (53760 // 2, 77824 // 2), "size": (2, 1), "level": 4},
    np.array([[[218], [237]], [[211], [230]], [[219], [237]]]),
]

TEST_CASE_2 = [
    filename,
    {"location": (53760 // 2, 77824 // 2), "size": (2, 1), "level": 2},
    np.array([[[229], [226]], [[218], [221]], [[232], [228]]]),
]

TEST_CASE_3 = [
    filename,
    {"location": (53760 // 2, 77824 // 2), "size": (8, 8), "level": 2, "grid_shape": (2, 1), "patch_size": 2},
    np.array(
        [
            [[[227, 228], [227, 228]], [[226, 228], [226, 228]], [[231, 228], [231, 230]]],
            [[[224, 224], [224, 226]], [[227, 228], [227, 227]], [[232, 231], [232, 231]]],
        ]
    ),
]

TEST_CASE_4 = [
    filename,
    {"location": (53760 // 2, 77824 // 2), "size": (8, 8), "level": 2, "grid_shape": (2, 1), "patch_size": 1},
    np.array([[[[228]], [[228]], [[230]]], [[[226]], [[227]], [[231]]]]),
]


class TestOpenSlideReader(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0])
    @skip_if_quick
    def test_read_whole_image(self, filename, expected_shape):
        self.camelyon_data_download(filename)
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj)

        self.assertTupleEqual(img.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @skip_if_quick
    def test_read_region(self, filename, patch_info, expected_img):
        self.camelyon_data_download(filename)
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj, **patch_info)
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    @parameterized.expand([TEST_CASE_3, TEST_CASE_4])
    @skip_if_quick
    def test_read_patches(self, filename, patch_info, expected_img):
        self.camelyon_data_download(filename)
        reader = WSIReader("OpenSlide")
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj, **patch_info)
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    def camelyon_data_download(self, filename):
        if not os.path.exists(filename):
            print(f"Test image [{filename}] does not exist. Downloading...")
            path = "gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/"
            ftp = ftplib.FTP("parrot.genomics.cn")
            ftp.login("anonymous", "")
            ftp.cwd(path)
            ftp.retrbinary("RETR " + filename, open(filename, "wb").write)
            ftp.quit()


if __name__ == "__main__":
    unittest.main()
