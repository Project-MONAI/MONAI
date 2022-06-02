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

from monai.data import DataLoader, Dataset
from monai.data.wsi_reader import WSIReader
from monai.transforms import Compose, LoadImaged, ToTensord
from monai.utils import first, optional_import
from monai.utils.enums import PostFix
from tests.utils import download_url_or_skip_test, testing_data_config, assert_allclose

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

HEIGHT = 32914
WIDTH = 46000

TEST_CASE_0 = [FILE_PATH, 2, (3, HEIGHT // 4, WIDTH // 4)]

TEST_CASE_TRANSFORM_0 = [FILE_PATH, 4, (HEIGHT // 16, WIDTH // 16), (1, 3, HEIGHT // 16, WIDTH // 16)]

TEST_CASE_1 = [
    FILE_PATH,
    {},
    {"location": (HEIGHT // 2, WIDTH // 2), "size": (2, 1), "level": 0},
    np.array([[[246], [246]], [[246], [246]], [[246], [246]]]),
]

TEST_CASE_2 = [
    FILE_PATH,
    {},
    {"location": (0, 0), "size": (2, 1), "level": 2},
    np.array([[[239], [239]], [[239], [239]], [[239], [239]]]),
]

TEST_CASE_3 = [
    FILE_PATH,
    {"channel_dim": -1},
    {"location": (HEIGHT // 2, WIDTH // 2), "size": (2, 1), "level": 0},
    np.moveaxis(np.array([[[246], [246]], [[246], [246]], [[246], [246]]]), 0, -1),
]

TEST_CASE_4 = [
    FILE_PATH,
    {"channel_dim": 2},
    {"location": (0, 0), "size": (2, 1), "level": 2},
    np.moveaxis(np.array([[[239], [239]], [[239], [239]], [[239], [239]]]), 0, -1),
]

TEST_CASE_MULTI_WSI = [
    [FILE_PATH, FILE_PATH],
    {"location": (0, 0), "size": (2, 1), "level": 2},
    np.concatenate(
        [
            np.array([[[239], [239]], [[239], [239]], [[239], [239]]]),
            np.array([[[239], [239]], [[239], [239]], [[239], [239]]]),
        ],
        axis=0,
    ),
]


TEST_CASE_RGB_0 = [np.ones((3, 2, 2), dtype=np.uint8)]  # CHW

TEST_CASE_RGB_1 = [np.ones((3, 100, 100), dtype=np.uint8)]  # CHW

TEST_CASE_ERROR_0C = [np.ones((16, 16), dtype=np.uint8)]  # no color channel
TEST_CASE_ERROR_1C = [np.ones((16, 16, 1), dtype=np.uint8)]  # one color channel
TEST_CASE_ERROR_2C = [np.ones((16, 16, 2), dtype=np.uint8)]  # two color channels
TEST_CASE_ERROR_3D = [np.ones((16, 16, 16, 3), dtype=np.uint8)]  # 3D + color


def save_rgba_tiff(array: np.ndarray, filename: str, mode: str):
    """
    Save numpy array into a TIFF RGB/RGBA file

    Args:
        array: numpy ndarray with the shape of CxHxW and C==3 representing a RGB image
        filename: the filename to be used for the tiff file. '_RGB.tiff' or '_RGBA.tiff' will be appended to this filename.
        mode: RGB or RGBA
    """
    if mode == "RGBA":
        array = np.concatenate([array, 255 * np.ones_like(array[0])[np.newaxis]]).astype(np.uint8)

    img_rgb = array.transpose(1, 2, 0)
    imwrite(filename, img_rgb, shape=img_rgb.shape, tile=(16, 16))

    return filename


def save_gray_tiff(array: np.ndarray, filename: str):
    """
    Save numpy array into a TIFF file

    Args:
        array: numpy ndarray with any shape
        filename: the filename to be used for the tiff file.
    """
    img_gray = array
    imwrite(filename, img_gray, shape=img_gray.shape)

    return filename


@skipUnless(has_cucim or has_osl or has_tiff, "Requires cucim, openslide, or tifffile!")
def setUpModule():  # noqa: N802
    hash_type = testing_data_config("images", FILE_KEY, "hash_type")
    hash_val = testing_data_config("images", FILE_KEY, "hash_val")
    download_url_or_skip_test(FILE_URL, FILE_PATH, hash_type=hash_type, hash_val=hash_val)


class WSIReaderTests:
    class Tests(unittest.TestCase):
        backend = None

        @parameterized.expand([TEST_CASE_0])
        def test_read_whole_image(self, file_path, level, expected_shape):
            reader = WSIReader(self.backend, level=level)
            with reader.read(file_path) as img_obj:
                img, meta = reader.get_data(img_obj)
            self.assertTupleEqual(img.shape, expected_shape)
            self.assertEqual(meta["backend"], self.backend)
            self.assertEqual(meta["path"], str(os.path.abspath(file_path)))
            self.assertEqual(meta["patch_level"], level)
            assert_array_equal(meta["patch_size"], expected_shape[1:])
            assert_array_equal(meta["patch_location"], (0, 0))

        @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
        def test_read_region(self, file_path, kwargs, patch_info, expected_img):
            reader = WSIReader(self.backend, **kwargs)
            with reader.read(file_path) as img_obj:
                if self.backend == "tifffile":
                    with self.assertRaises(ValueError):
                        reader.get_data(img_obj, **patch_info)[0]
                else:
                    # Read twice to check multiple calls
                    img, meta = reader.get_data(img_obj, **patch_info)
                    img2 = reader.get_data(img_obj, **patch_info)[0]
                    self.assertTupleEqual(img.shape, img2.shape)
                    self.assertIsNone(assert_array_equal(img, img2))
                    self.assertTupleEqual(img.shape, expected_img.shape)
                    self.assertIsNone(assert_array_equal(img, expected_img))
                    self.assertEqual(meta["backend"], self.backend)
                    self.assertEqual(meta["path"], str(os.path.abspath(file_path)))
                    self.assertEqual(meta["patch_level"], patch_info["level"])
                    assert_array_equal(meta["patch_size"], patch_info["size"])
                    assert_array_equal(meta["patch_location"], patch_info["location"])

        @parameterized.expand([TEST_CASE_MULTI_WSI])
        def test_read_region_multi_wsi(self, file_path_list, patch_info, expected_img):
            kwargs = {"name": None, "offset": None} if self.backend == "tifffile" else {}
            reader = WSIReader(self.backend, **kwargs)
            img_obj_list = reader.read(file_path_list, **kwargs)
            if self.backend == "tifffile":
                with self.assertRaises(ValueError):
                    reader.get_data(img_obj_list, **patch_info)[0]
            else:
                # Read twice to check multiple calls
                img, meta = reader.get_data(img_obj_list, **patch_info)
                img2 = reader.get_data(img_obj_list, **patch_info)[0]
                self.assertTupleEqual(img.shape, img2.shape)
                self.assertIsNone(assert_array_equal(img, img2))
                self.assertTupleEqual(img.shape, expected_img.shape)
                self.assertIsNone(assert_array_equal(img, expected_img))
                self.assertEqual(meta["backend"], self.backend)
                self.assertEqual(meta["path"][0], str(os.path.abspath(file_path_list[0])))
                self.assertEqual(meta["patch_level"][0], patch_info["level"])
                assert_array_equal(meta["patch_size"][0], expected_img.shape[1:])
                assert_array_equal(meta["patch_location"][0], patch_info["location"])

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
                with reader.read(file_path) as img_obj:
                    image[mode], _ = reader.get_data(img_obj)

            self.assertIsNone(assert_array_equal(image["RGB"], img_expected))
            self.assertIsNone(assert_array_equal(image["RGBA"], img_expected))

        @parameterized.expand([TEST_CASE_ERROR_0C, TEST_CASE_ERROR_1C, TEST_CASE_ERROR_2C, TEST_CASE_ERROR_3D])
        @skipUnless(has_tiff, "Requires tifffile.")
        def test_read_malformats(self, img_expected):
            if self.backend == "cucim" and (len(img_expected.shape) < 3 or img_expected.shape[2] == 1):
                # Until cuCIM addresses https://github.com/rapidsai/cucim/issues/230
                return
            reader = WSIReader(self.backend)
            file_path = os.path.join(os.path.dirname(__file__), "testing_data", "temp_tiff_image_gray.tiff")
            imwrite(file_path, img_expected, shape=img_expected.shape)
            with self.assertRaises((RuntimeError, ValueError, openslide.OpenSlideError if has_osl else ValueError)):
                with reader.read(file_path) as img_obj:
                    reader.get_data(img_obj)

        @parameterized.expand([TEST_CASE_TRANSFORM_0])
        def test_with_dataloader(self, file_path, level, expected_spatial_shape, expected_shape):
            train_transform = Compose(
                [
                    LoadImaged(keys=["image"], reader=WSIReader, backend=self.backend, level=level),
                    ToTensord(keys=["image"]),
                ]
            )
            dataset = Dataset([{"image": file_path}], transform=train_transform)
            data_loader = DataLoader(dataset)
            data: dict = first(data_loader)
            for s in data[PostFix.meta("image")]["spatial_shape"]:
                assert_allclose(s, expected_spatial_shape, type_test=False)
            self.assertTupleEqual(data["image"].shape, expected_shape)

        @parameterized.expand([TEST_CASE_TRANSFORM_0])
        def test_with_dataloader_batch(self, file_path, level, expected_spatial_shape, expected_shape):
            train_transform = Compose(
                [
                    LoadImaged(keys=["image"], reader=WSIReader, backend=self.backend, level=level),
                    ToTensord(keys=["image"]),
                ]
            )
            dataset = Dataset([{"image": file_path}, {"image": file_path}], transform=train_transform)
            batch_size = 2
            data_loader = DataLoader(dataset, batch_size=batch_size)
            data: dict = first(data_loader)
            for s in data[PostFix.meta("image")]["spatial_shape"]:
                assert_allclose(s, expected_spatial_shape, type_test=False)
            self.assertTupleEqual(data["image"].shape, (batch_size, *expected_shape[1:]))


@skipUnless(has_cucim, "Requires cucim")
class TestCuCIM(WSIReaderTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "cucim"


@skipUnless(has_osl, "Requires openslide")
class TestOpenSlide(WSIReaderTests.Tests):
    @classmethod
    def setUpClass(cls):
        cls.backend = "openslide"


if __name__ == "__main__":
    unittest.main()
