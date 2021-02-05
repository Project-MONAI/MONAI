import ftplib
import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.data.image_reader import CuImageReader
from tests.utils import skip_if_quick

filename = "test_001.tif"


TEST_CASE_1 = [
    filename,
    {"location": (86016 // 2, 89600 // 2), "size": (1, 2), "level": 4},
    np.array([[[234], [223]], [[174], [163]], [[228], [217]]]),
]

TEST_CASE_2 = [
    filename,
    {"location": (86016 // 2, 89600 // 2), "size": (1, 2), "level": 2},
    np.array([[[220], [197]], [[165], [143]], [[220], [195]]]),
]

TEST_CASE_3 = [
    filename,
    {"location": (86016 // 2, 89600 // 2), "size": (8, 8), "level": 2, "grid_shape": (2, 1), "patch_size": 2},
    np.array(
        [
            [[[218, 242], [189, 198]], [[154, 173], [125, 132]], [[214, 236], [185, 194]]],
            [[[190, 209], [221, 228]], [[120, 137], [149, 154]], [[180, 200], [212, 217]]],
        ]
    ),
]

TEST_CASE_4 = [
    filename,
    {"location": (86016 // 2, 89600 // 2), "size": (8, 8), "level": 2, "grid_shape": (2, 1), "patch_size": 1},
    np.array(
        [
            [[[198]], [[132]], [[194]]],
            [[[228]], [[154]], [[217]]]
        ]
    ),
]

class TestCuImageReader(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @skip_if_quick
    def test_read_region(self, filename, patch_info, expected_img):
        self.camelyon_data_download(filename)
        reader = CuImageReader()
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj, **patch_info)

        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    @parameterized.expand([TEST_CASE_3, TEST_CASE_4])
    @skip_if_quick
    def test_read_patches(self, filename, patch_info, expected_img):
        self.camelyon_data_download(filename)
        reader = CuImageReader()
        img_obj = reader.read(filename)
        img = reader.get_data(img_obj, **patch_info)
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    def camelyon_data_download(self, filename):
        if not os.path.exists(filename):
            print(f"Test image [{filename}] does not exists downloading...")
            path = "gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/"
            ftp = ftplib.FTP("parrot.genomics.cn")
            ftp.login("anonymous", "")
            ftp.cwd(path)
            ftp.retrbinary("RETR " + filename, open(filename, "wb").write)
            ftp.quit()


if __name__ == "__main__":
    unittest.main()
