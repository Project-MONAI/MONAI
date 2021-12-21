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

from monai.transforms import AdjustContrastd
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

TEST_CASE_1 = [1.0]

TEST_CASE_2 = [0.5]

TEST_CASE_3 = [4.5]


class TestAdjustContrastd(NumpyImageTestCase2D):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_correct_results(self, gamma):
        adjuster = AdjustContrastd("img", gamma=gamma)
        for p in TEST_NDARRAYS:
            result = adjuster({"img": p(self.imt)})
            if gamma == 1.0:
                expected = self.imt
            else:
                epsilon = 1e-7
                img_min = self.imt.min()
                img_range = self.imt.max() - img_min
                expected = np.power(((self.imt - img_min) / float(img_range + epsilon)), gamma) * img_range + img_min
            assert_allclose(expected, result["img"], rtol=1e-05, type_test=False)


if __name__ == "__main__":
    unittest.main()
