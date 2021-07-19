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

import unittest

import numpy as np
from parameterized import parameterized

import monai
from monai.transforms import BoundingRectD
from tests.utils import TEST_NDARRAYS

SEED = 1

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p, (2, 3), [[0, 0], [1, 2]]])
    TESTS.append([p, (1, 8, 10), [[0, 7, 1, 9]]])
    TESTS.append([p, (2, 16, 20, 18), [[0, 16, 0, 20, 0, 18], [0, 16, 0, 20, 0, 18]]])


class TestBoundingRectD(unittest.TestCase):
    def setUp(self):
        monai.utils.set_determinism(SEED)

    def tearDown(self):
        monai.utils.set_determinism(None)

    @parameterized.expand(TESTS)
    def test_value(self, im_type, input_shape, expected):
        np.random.seed(SEED)
        test_data = im_type(np.random.randint(0, 8, size=input_shape))
        test_data = test_data == 7
        result = BoundingRectD("image")({"image": test_data})
        np.testing.assert_allclose(result["image_bbox"], expected)

        result = BoundingRectD("image", "cc")({"image": test_data})
        np.testing.assert_allclose(result["image_cc"], expected)

        with self.assertRaises(KeyError):
            BoundingRectD("image", "cc")({"image": test_data, "image_cc": None})


if __name__ == "__main__":
    unittest.main()
