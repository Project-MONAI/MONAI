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

import itk
import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import ITKReader
from monai.transforms import LoadImaged

KEYS = ["image", "label", "extra"]

TEST_CASE_1 = [{"keys": KEYS}, (128, 128, 128)]

TEST_CASE_2 = [{"keys": KEYS, "reader": "ITKReader", "fallback_only": False}, (128, 128, 128)]


class TestLoadImaged(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, expected_shape):
        test_image = nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4))
        test_data = {}
        with tempfile.TemporaryDirectory() as tempdir:
            for key in KEYS:
                nib.save(test_image, os.path.join(tempdir, key + ".nii.gz"))
                test_data.update({key: os.path.join(tempdir, key + ".nii.gz")})
            result = LoadImaged(**input_param)(test_data)

        for key in KEYS:
            self.assertTupleEqual(result[key].shape, expected_shape)

    def test_register(self):
        spatial_size = (32, 64, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImaged(keys="img")
            loader.register(ITKReader())
            result = loader({"img": filename})
            self.assertTupleEqual(tuple(result["img_meta_dict"]["spatial_shape"]), spatial_size)
            self.assertTupleEqual(result["img"].shape, spatial_size)


if __name__ == "__main__":
    unittest.main()
