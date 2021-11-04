# Copyright 2020 - 2021 MONAI Consortium
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

from monai.apps.utils import download_url
from monai.data import DataLoader, Dataset
from monai.data.image_reader import WSIReader
from monai.transforms import Compose, LoadImaged, ToTensord
from monai.utils import first, optional_import

cucim, has_cucim = optional_import("cucim")
has_cucim = has_cucim and hasattr(cucim, "CuImage")
_, has_osl = optional_import("openslide")
imsave, has_tiff = optional_import("tifffile", name="imsave")

FILE_URL = "https://drive.google.com/uc?id=1sGTKZlJBIz53pfqTxoTqiIQzIoEzHLAe"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + os.path.basename(FILE_URL))

HEIGHT = 32914
WIDTH = 46000

TEST_CASE_0 = [FILE_PATH, 2, (3, HEIGHT // 4, WIDTH // 4)]

TEST_CASE_TRANSFORM_0 = [FILE_PATH, 4, (HEIGHT // 16, WIDTH // 16), (1, 3, HEIGHT // 16, WIDTH // 16)]

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
    {"location": (0, 0), "size": (8, 8), "level": 2, "grid_shape": (2, 1), "patch_size": 2},
    np.array(
        [
            [[[239, 239], [239, 239]], [[239, 239], [239, 239]], [[239, 239], [239, 239]]],
            [[[242, 242], [242, 243]], [[242, 242], [242, 243]], [[242, 242], [242, 243]]],
        ]
    ),
]

TEST_CASE_4 = [
    FILE_PATH,
    {"location": (0, 0), "size": (8, 8), "level": 2, "grid_shape": (2, 1), "patch_size": 1},
    np.array([[[[239]], [[239]], [[239]]], [[[243]], [[243]], [[243]]]]),
]

TEST_CASE_RGB_0 = [np.ones((3, 2, 2), dtype=np.uint8)]  # CHW

TEST_CASE_RGB_1 = [np.ones((3, 100, 100), dtype=np.uint8)]  # CHW


def save_rgba_tiff(array: np.ndarray, filename: str, mode: str):
    """
    Save numpy array into a TIFF RGB/RGBA file

    Args:
        array: numpy ndarray with the shape of CxHxW and C==3 representing a RGB image
        file_prefix: the filename to be used for the tiff file. '_RGB.tiff' or '_RGBA.tiff' will be appended to this filename.
        mode: RGB or RGBA
    """
    if mode == "RGBA":
        array = np.concatenate([array, 255 * np.ones_like(array[0])[np.newaxis]]).astype(np.uint8)

    img_rgb = array.transpose(1, 2, 0)
    imsave(filename, img_rgb, shape=img_rgb.shape, tile=(16, 16))

    return filename


@skipUnless(has_cucim or has_osl, "Requires cucim or openslide!")
def setUpModule():  # noqa: N802
    download_url(FILE_URL, FILE_PATH, "5a3cfd4fd725c50578ddb80b517b759f")


class WSIReaderTests:
    backend = None

    @parameterized.expand([TEST_CASE_0])
    def test_read_whole_image(self, file_path, level, expected_shape):
        reader = WSIReader(self.backend, level=level)
        img_obj = reader.read(file_path)
        img = reader.get_data(img_obj)[0]
        self.assertTupleEqual(img.shape, expected_shape)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_read_region(self, file_path, patch_info, expected_img):
        reader = WSIReader(self.backend)
        img_obj = reader.read(file_path)
        img = reader.get_data(img_obj, **patch_info)[0]
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    @parameterized.expand([TEST_CASE_3, TEST_CASE_4])
    def test_read_patches(self, file_path, patch_info, expected_img):
        reader = WSIReader(self.backend)
        img_obj = reader.read(file_path)
        img = reader.get_data(img_obj, **patch_info)[0]
        self.assertTupleEqual(img.shape, expected_img.shape)
        self.assertIsNone(assert_array_equal(img, expected_img))

    @parameterized.expand([TEST_CASE_RGB_0, TEST_CASE_RGB_1])
    @skipUnless(has_tiff, "Requires tifffile.")
    def test_read_rgba(self, img_expected):
        # skip for OpenSlide since not working with images without tiles
        if self.backend == "openslide":
            return
        image = {}
        reader = WSIReader(self.backend)
        for mode in ["RGB", "RGBA"]:
            file_path = save_rgba_tiff(
                img_expected,
                os.path.join(os.path.dirname(__file__), "testing_data", f"temp_tiff_image_{mode}.tiff"),
                mode=mode,
            )
            img_obj = reader.read(file_path)
            image[mode], _ = reader.get_data(img_obj)

        self.assertIsNone(assert_array_equal(image["RGB"], img_expected))
        self.assertIsNone(assert_array_equal(image["RGBA"], img_expected))

    @parameterized.expand([TEST_CASE_TRANSFORM_0])
    def test_with_dataloader(self, file_path, level, expected_spatial_shape, expected_shape):
        train_transform = Compose(
            [
                LoadImaged(keys=["image"], reader=WSIReader, backend="cuCIM", level=level),
                ToTensord(keys=["image"]),
            ]
        )
        dataset = Dataset([{"image": file_path}], transform=train_transform)
        data_loader = DataLoader(dataset)
        data: dict = first(data_loader)
        spatial_shape = tuple(d.item() for d in data["image_meta_dict"]["spatial_shape"])
        self.assertTupleEqual(spatial_shape, expected_spatial_shape)
        self.assertTupleEqual(data["image"].shape, expected_shape)


@skipUnless(has_cucim, "Requires cucim")
class TestCuCIM(unittest.TestCase, WSIReaderTests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires OpenSlide")
class TestOpenSlide(unittest.TestCase, WSIReaderTests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


if __name__ == "__main__":
    unittest.main()
