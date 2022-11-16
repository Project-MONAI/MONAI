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

from monai.transforms import AdjustContrast
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

TEST_CASE_1 = [1.0, None]

TEST_CASE_2 = [0.5, None]

TEST_CASE_3 = [4.5, None]

TEST_CASE_4 = [4.5, [0.0, 1.0]]


class TestAdjustContrast(NumpyImageTestCase2D):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_correct_results(self, gamma, clip_range):
        adjuster = AdjustContrast(gamma=gamma, clip_range=clip_range)
        for p in TEST_NDARRAYS:
            im = p(self.imt)
            result = adjuster(im)
            self.assertTrue(type(im), type(result))
            if gamma == 1.0:
                expected = self.imt
            else:
                epsilon = 1e-7
                img_min = self.imt.min()
                img_range = self.imt.max() - img_min
                if clip_range is not None:
                    expected = np.clip(
                        (np.power(((self.imt - img_min) / float(img_range + epsilon)), gamma) * img_range + img_min),
                        0.0,
                        1.0,
                    )
                else:
                    expected = (
                        np.power(((self.imt - img_min) / float(img_range + epsilon)), gamma) * img_range + img_min
                    )
            assert_allclose(result, expected, rtol=1e-05, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
