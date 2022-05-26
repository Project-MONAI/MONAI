# Copyright (c) MONAI Consortium
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

from monai.transforms.intensity.array import ForegroundMask
from monai.utils import set_determinism

set_determinism(1234)

A = np.random.randint(51, 128, (3, 3, 2))
B = np.ones_like(A[:1])
IMAGE1 = np.pad(A, ((0, 0), (2, 2), (2, 2)), constant_values=255)
IMAGE2 = np.copy(IMAGE1)
IMAGE2[0] = 0
IMAGE3 = np.copy(IMAGE2)
IMAGE3[1] = 0
MASK = np.pad(B, ((0, 0), (2, 2), (2, 2)), constant_values=0)
TEST_CASE_1 = [{"threshold": "otsu"}, IMAGE1, MASK]
TEST_CASE_2 = [{"threshold": "otsu"}, IMAGE2, MASK]
TEST_CASE_3 = [{"threshold": "otsu"}, IMAGE3, MASK]


class TestForegroundMask(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_3])
    def test_foreground_mask(self, arguments, image, mask):
        result = ForegroundMask(**arguments)(image)
        np.testing.assert_allclose(result, mask)


if __name__ == "__main__":
    unittest.main()
