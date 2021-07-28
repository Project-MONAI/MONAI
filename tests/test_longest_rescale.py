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
import skimage.transform
from parameterized import parameterized

from monai.transforms import LongestRescale

TEST_CASE_0 = [{"spatial_size": 15}, (6, 11, 15)]

TEST_CASE_1 = [{"spatial_size": 15, "mode": "area"}, (6, 11, 15)]

TEST_CASE_2 = [{"spatial_size": 6, "mode": "trilinear", "align_corners": True}, (3, 5, 6)]


class TestLongestRescale(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, expected_shape):
        input_data = np.random.randint(0, 2, size=[3, 4, 7, 10])
        result = LongestRescale(**input_param)(input_data)
        np.testing.assert_allclose(result.shape[1:], expected_shape)


if __name__ == "__main__":
    unittest.main()
