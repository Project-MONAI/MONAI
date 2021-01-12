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

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.transforms import LoadNifti

TEST_CASE_1 = [{"as_closest_canonical": False, "image_only": True}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_2 = [{"as_closest_canonical": False, "image_only": False}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_3 = [
    {"as_closest_canonical": False, "image_only": True},
    ["test_image1.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_4 = [
    {"as_closest_canonical": False, "image_only": False},
    ["test_image1.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_5 = [{"as_closest_canonical": True, "image_only": False}, ["test_image.nii.gz"], (128, 128, 128)]


class TestLoadNifti(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_shape(self, input_param, filenames, expected_shape):
        test_image = np.random.randint(0, 2, size=[128, 128, 128])
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])
            result = LoadNifti(**input_param)(filenames)

        if isinstance(result, tuple):
            result, header = result
            self.assertTrue("affine" in header)
            np.testing.assert_allclose(header["affine"], np.eye(4))
            if input_param["as_closest_canonical"]:
                np.testing.assert_allclose(header["original_affine"], np.eye(4))
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
