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
from monai.transforms.composables import LoadNiftid

TEST_CASE_1 = [
    {
        'keys': ['image', 'label', 'extra'],
        'as_closest_canonical': False
    },
    {
        'image': 'test_image.nii.gz',
        'label': 'test_label.nii.gz',
        'extra': 'test_extra.nii.gz'
    },
    (128, 128, 128)
]


class TestLoadNiftid(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, input_param, input_data, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]), np.eye(4))
        nib.save(test_image, 'test_image.nii.gz')
        nib.save(test_image, 'test_label.nii.gz')
        nib.save(test_image, 'test_extra.nii.gz')
        result = LoadNiftid(**input_param)(input_data)
        if os.path.exists('test_image.nii.gz'):
            os.remove('test_image.nii.gz')
        if os.path.exists('test_label.nii.gz'):
            os.remove('test_label.nii.gz')
        if os.path.exists('test_extra.nii.gz'):
            os.remove('test_extra.nii.gz')
        self.assertTupleEqual(result['image'].shape, expected_shape)
        self.assertTupleEqual(result['label'].shape, expected_shape)
        self.assertTupleEqual(result['extra'].shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
