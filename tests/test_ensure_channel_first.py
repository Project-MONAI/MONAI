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
import tempfile
import unittest

import itk
import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized
from PIL import Image

from monai.data import ITKReader
from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirst, LoadImage

TEST_CASE_1 = [{}, ["test_image.nii.gz"], None]

TEST_CASE_2 = [{}, ["test_image.nii.gz"], -1]

TEST_CASE_3 = [{}, ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"], None]

TEST_CASE_4 = [{"reader": ITKReader()}, ["test_image.nii.gz"], None]

TEST_CASE_5 = [{"reader": ITKReader()}, ["test_image.nii.gz"], -1]

TEST_CASE_6 = [{"reader": ITKReader()}, ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"], None]

TEST_CASE_7 = [{"reader": ITKReader(pixel_type=itk.UC)}, "tests/testing_data/CT_DICOM", None]


class TestEnsureChannelFirst(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_load_nifti(self, input_param, filenames, original_channel_dim):
        if original_channel_dim is None:
            test_image = np.random.rand(128, 128, 128)
        elif original_channel_dim == -1:
            test_image = np.random.rand(128, 128, 128, 1)

        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])

            result = LoadImage(image_only=True, **input_param)(filenames)
            result = EnsureChannelFirst()(result)
            self.assertEqual(result.shape[0], len(filenames))

    @parameterized.expand([TEST_CASE_7])
    def test_itk_dicom_series_reader(self, input_param, filenames, _):
        result = LoadImage(image_only=True, **input_param)(filenames)
        result = EnsureChannelFirst()(result)
        self.assertEqual(result.shape[0], 1)

    def test_load_png(self):
        spatial_size = (256, 256, 3)
        test_image = np.random.randint(0, 256, size=spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            Image.fromarray(test_image.astype("uint8")).save(filename)
            result = LoadImage(image_only=True)(filename)
            result = EnsureChannelFirst()(result)
            self.assertEqual(result.shape[0], 3)

    def test_check(self):
        im = torch.zeros(1, 2, 3)
        with self.assertRaises(ValueError):  # not MetaTensor
            EnsureChannelFirst()(im)
        with self.assertRaises(ValueError):  # no meta
            EnsureChannelFirst()(MetaTensor(im))
        with self.assertRaises(ValueError):  # no meta channel
            EnsureChannelFirst()(MetaTensor(im, meta={"original_channel_dim": None}))
        EnsureChannelFirst(strict_check=False)(im)
        EnsureChannelFirst(strict_check=False)(MetaTensor(im, meta={"original_channel_dim": None}))


if __name__ == "__main__":
    unittest.main()
