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
from monai.transforms import LoadPNGd

KEYS = ["image", "label", "extra"]

TEST_CASE_1 = [{"keys": KEYS}, (128, 128, 3)]


class TestLoadPNGd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, input_param, expected_shape):
        test_image = np.random.randint(0, 256, size=[128, 128, 3])
        tempdir = tempfile.mkdtemp()
        test_data = dict()
        for key in KEYS:
            Image.fromarray(test_image.astype("uint8")).save(os.path.join(tempdir, key + ".png"))
            test_data.update({key: os.path.join(tempdir, key + ".png")})
        result = LoadPNGd(**input_param)(test_data)
        for key in KEYS:
            self.assertTupleEqual(result[key].shape, expected_shape)
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
