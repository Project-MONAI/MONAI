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
import shutil
import numpy as np
import tempfile
import nibabel as nib
from parameterized import parameterized
from monai.transforms.transforms import LoadNifti

TEST_CASE_IMAGE_ONLY = [
    {
        'as_closest_canonical': False,
        'image_only': True
    },
    (128, 128, 128)
]

TEST_CASE_IMAGE_METADATA = [
    {
        'as_closest_canonical': False,
        'image_only': False
    },
    (128, 128, 128)
]


class TestLoadNifti(unittest.TestCase):

    @parameterized.expand([TEST_CASE_IMAGE_ONLY, TEST_CASE_IMAGE_METADATA])
    def test_shape(self, input_param, expected_shape):
        test_image = np.random.randint(0, 2, size=[128, 128, 128])
        tempdir = tempfile.mkdtemp()
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), os.path.join(tempdir, 'test_image.nii.gz'))
        test_data = os.path.join(tempdir, 'test_image.nii.gz')
        result = LoadNifti(**input_param)(test_data)
        shutil.rmtree(tempdir)
        if isinstance(result, tuple):
            result = result[0]
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
