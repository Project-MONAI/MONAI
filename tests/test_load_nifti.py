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

import unittest
import os
import numpy as np
import nibabel as nib
from parameterized import parameterized
from monai.transforms.transforms import LoadNifti

TEST_CASE_1 = [
    {
        'as_closest_canonical': False,
        'image_only': True
    },
    'test_image.nii.gz',
    (128, 128, 128)
]

TEST_CASE_2 = [
    {
        'as_closest_canonical': False,
        'image_only': False
    },
    'test_image.nii.gz',
    (128, 128, 128)
]


class TestLoadNifti(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_shape):
        test_image = np.random.randint(0, 2, size=[128, 128, 128])
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), 'test_image.nii.gz')
        result = LoadNifti(**input_param)(input_data)
        if os.path.exists('test_image.nii.gz'):
            os.remove('test_image.nii.gz')
        if isinstance(result, tuple):
            result = result[0]
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
