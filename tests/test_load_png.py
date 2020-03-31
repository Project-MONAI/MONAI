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
import tempfile
from PIL import Image
from parameterized import parameterized
from monai.transforms.transforms import LoadPNG

TEST_CASE_1 = [
    (128, 128),
    ['test_image.png'],
    (128, 128)
]

TEST_CASE_2 = [
    (128, 128, 3),
    ['test_image.png'],
    (128, 128, 3)
]

TEST_CASE_3 = [
    (128, 128),
    ['test_image1.png', 'test_image2.png', 'test_image3.png'],
    (3, 128, 128)
]


class TestLoadPNG(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, data_shape, filenames, expected_shape):
        test_image = np.random.randint(0, 256, size=data_shape)
        tempdir = tempfile.mkdtemp()
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                Image.fromarray(test_image.astype('uint8')).save(filenames[i])
            result = LoadPNG()(filenames)
        if isinstance(result, tuple):
            result = result[0]
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
