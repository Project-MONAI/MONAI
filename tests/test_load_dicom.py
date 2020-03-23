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
import pydicom
from pydicom.data import get_testdata_files
from parameterized import parameterized
from monai.transforms.transforms import LoadDICOM


TEST_CASE_IMAGE_ONLY = [
    {'image_only': True}
]

TEST_CASE_IMAGE_METADATA = [
    {'image_only': False}
]


class TestLoadDICOM(unittest.TestCase):

    @parameterized.expand([TEST_CASE_IMAGE_ONLY, TEST_CASE_IMAGE_METADATA])
    def test_shape(self, input_param):
        filename = get_testdata_files('CT_small.dcm')[0]
        dataset = pydicom.dcmread(filename)
        expected_shape = dataset.pixel_array.shape
        result = LoadDICOM(**input_param)(filename)
        if isinstance(result, tuple):
            result = result[0]
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
