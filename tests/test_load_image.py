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
import tempfile
import unittest
from pathlib import Path

import itk
import nibabel as nib
import numpy as np
from parameterized import parameterized
from PIL import Image

from monai.data import ITKReader, NibabelReader
from monai.transforms import SUPPORTED_READERS, LoadImage


class _MiniReader:
    """a test case customised reader"""

    def __init__(self, is_compatible=False):
        self.is_compatible = is_compatible

    def verify_suffix(self, _name):
        return self.is_compatible

    def read(self, name):
        return name

    def get_data(self, _obj):
        return np.zeros((1, 1, 1)), {"name": "my test"}


TEST_CASE_1 = [{"image_only": True}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_2 = [{"image_only": False}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_3 = [
    {"image_only": True},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_3_1 = [  # .mgz format
    {"image_only": True, "reader": "nibabelreader"},
    ["test_image.mgz", "test_image2.mgz", "test_image3.mgz"],
    (3, 128, 128, 128),
]

TEST_CASE_4 = [
    {"image_only": False},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_4_1 = [  # additional parameter
    {"image_only": False, "mmap": False},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_5 = [
    {"reader": NibabelReader(mmap=False), "image_only": False},
    ["test_image.nii.gz"],
    (128, 128, 128),
]

TEST_CASE_6 = [{"reader": ITKReader(), "image_only": True}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_7 = [{"reader": ITKReader(), "image_only": False}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_8 = [
    {"reader": ITKReader(), "image_only": True},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_9 = [
    {"reader": ITKReader(), "image_only": False},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_10 = [
    {"image_only": False, "reader": ITKReader(pixel_type=itk.UC)},
    "tests/testing_data/CT_DICOM",
    (16, 16, 4),
]

TEST_CASE_11 = [
    {"image_only": False, "reader": "ITKReader", "pixel_type": itk.UC},
    "tests/testing_data/CT_DICOM",
    (16, 16, 4),
]


class TestLoadImage(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_3_1, TEST_CASE_4, TEST_CASE_4_1, TEST_CASE_5]
    )
    def test_nibabel_reader(self, input_param, filenames, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])
            result = LoadImage(**input_param)(filenames)

            if isinstance(result, tuple):
                result, header = result
                self.assertTrue("affine" in header)
                self.assertEqual(header["filename_or_obj"], os.path.join(tempdir, "test_image.nii.gz"))
                np.testing.assert_allclose(header["affine"], np.eye(4))
                np.testing.assert_allclose(header["original_affine"], np.eye(4))
            self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_6, TEST_CASE_7, TEST_CASE_8, TEST_CASE_9])
    def test_itk_reader(self, input_param, filenames, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                itk_np_view = itk.image_view_from_array(test_image)
                itk.imwrite(itk_np_view, filenames[i])
            result = LoadImage(**input_param)(filenames)

            if isinstance(result, tuple):
                result, header = result
                self.assertTrue("affine" in header)
                self.assertEqual(header["filename_or_obj"], os.path.join(tempdir, "test_image.nii.gz"))
                np_diag = np.diag([-1, -1, 1, 1])
                np.testing.assert_allclose(header["affine"], np_diag)
                np.testing.assert_allclose(header["original_affine"], np_diag)
            self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_10, TEST_CASE_11])
    def test_itk_dicom_series_reader(self, input_param, filenames, expected_shape):
        result, header = LoadImage(**input_param)(filenames)
        self.assertTrue("affine" in header)
        self.assertEqual(header["filename_or_obj"], filenames)
        np.testing.assert_allclose(
            header["affine"],
            np.array(
                [
                    [-0.488281, 0.0, 0.0, 125.0],
                    [0.0, -0.488281, 0.0, 128.100006],
                    [0.0, 0.0, 68.33333333, -99.480003],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        self.assertTupleEqual(result.shape, expected_shape)
        self.assertTupleEqual(tuple(header["spatial_shape"]), expected_shape)

    def test_itk_reader_multichannel(self):
        test_image = np.random.randint(0, 256, size=(256, 224, 3)).astype("uint8")
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            itk_np_view = itk.image_view_from_array(test_image, is_vector=True)
            itk.imwrite(itk_np_view, filename)
            result, header = LoadImage(reader=ITKReader())(Path(filename))

            self.assertTupleEqual(tuple(header["spatial_shape"]), (224, 256))
            np.testing.assert_allclose(result[:, :, 0], test_image[:, :, 0].T)
            np.testing.assert_allclose(result[:, :, 1], test_image[:, :, 1].T)
            np.testing.assert_allclose(result[:, :, 2], test_image[:, :, 2].T)

    def test_load_png(self):
        spatial_size = (256, 224)
        test_image = np.random.randint(0, 256, size=spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            Image.fromarray(test_image.astype("uint8")).save(filename)
            result, header = LoadImage(image_only=False)(filename)
            self.assertTupleEqual(tuple(header["spatial_shape"]), spatial_size[::-1])
            self.assertTupleEqual(result.shape, spatial_size[::-1])
            np.testing.assert_allclose(result.T, test_image)

    def test_register(self):
        spatial_size = (32, 64, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImage(image_only=False)
            loader.register(ITKReader())
            result, header = loader(filename)
            self.assertTupleEqual(tuple(header["spatial_shape"]), spatial_size[::-1])
            self.assertTupleEqual(result.shape, spatial_size[::-1])

    def test_kwargs(self):
        spatial_size = (32, 64, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImage(image_only=False)
            reader = ITKReader(fallback_only=False)
            loader.register(reader)
            result, header = loader(filename)

            reader = ITKReader()
            img = reader.read(filename, fallback_only=False)
            result_raw, header_raw = reader.get_data(img)
            np.testing.assert_allclose(header["spatial_shape"], header_raw["spatial_shape"])
            self.assertTupleEqual(result.shape, result_raw.shape)

    def test_my_reader(self):
        """test customised readers"""
        out = LoadImage(reader=_MiniReader, is_compatible=True)("test")
        self.assertEqual(out[1]["name"], "my test")
        out = LoadImage(reader=_MiniReader, is_compatible=False)("test")
        self.assertEqual(out[1]["name"], "my test")
        SUPPORTED_READERS["minireader"] = _MiniReader
        for item in ("minireader", _MiniReader, _MiniReader(is_compatible=False)):
            out = LoadImage(reader=item)("test")
            self.assertEqual(out[1]["name"], "my test")
        out = LoadImage()("test", reader=_MiniReader(is_compatible=False))
        self.assertEqual(out[1]["name"], "my test")


if __name__ == "__main__":
    unittest.main()
