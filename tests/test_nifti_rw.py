# Copyright 2020 MONAI Consortium
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

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import load_nifti, write_nifti

from tests.utils import make_nifti_image

TEST_IMAGE = np.zeros((1, 2, 3))
TEST_AFFINE = np.array([[-5.3, 0., 0., 102.01], [0., 0.52, 2.17, -7.50], [-0., 1.98, -0.26, -23.12], [0., 0., 0., 1.]])

TEST_CASE_1 = [TEST_IMAGE, TEST_AFFINE, (1, 2, 3), dict(as_closest_canonical=True, image_only=False)]
TEST_CASE_2 = [TEST_IMAGE, TEST_AFFINE, (1, 3, 2), dict(as_closest_canonical=True, image_only=True)]
TEST_CASE_3 = [TEST_IMAGE, TEST_AFFINE, (1, 2, 3), dict(as_closest_canonical=False, image_only=True)]
TEST_CASE_4 = [TEST_IMAGE, TEST_AFFINE, (1, 2, 3), dict(as_closest_canonical=False, image_only=False)]


class TestNiftiLoadRead(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_orientation(self, array, affine, expected_shape, reader_param):
        test_image = make_nifti_image(array, affine)

        # read test cases
        load_result = load_nifti(test_image, **reader_param)
        if isinstance(load_result, tuple):
            data_array, header = load_result
        else:
            data_array = load_result
            header = None
        if os.path.exists(test_image):
            os.remove(test_image)

        # write test cases
        if header is not None:
            write_nifti(data_array, test_image, header['affine'], header['original_affine'])
        else:
            write_nifti(data_array, test_image, affine)
        saved = nib.load(test_image)
        saved_affine = saved.affine
        saved_shape = saved.get_fdata().shape
        if os.path.exists(test_image):
            os.remove(test_image)

        self.assertTrue(np.allclose(saved_affine, affine))
        self.assertTrue(np.allclose(saved_shape, expected_shape))


if __name__ == '__main__':
    unittest.main()
