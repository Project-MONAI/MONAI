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
from PIL import Image
from parameterized import parameterized
from monai.application import check_md5

TEST_CASE_1 = ["f38e9e043c8e902321e827b24ce2e5ec", True]

TEST_CASE_2 = ["12c730d4e7427e00ad1c5526a6677535", False]

TEST_CASE_3 = [None, True]


class TestCheckMD5(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, md5_value, expected_result):
        test_image = np.ones((64, 64, 3))
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, "test_file.png")
        Image.fromarray(test_image.astype("uint8")).save(filename)

        result = check_md5(filename, md5_value)
        self.assertTrue(result == expected_result)

        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
