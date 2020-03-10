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

import numpy as np
from parameterized import parameterized

from monai.transforms.transforms import Orientation

TEST_CASES = [
    [{'axcodes': 'RAS'},
     np.ones((2, 10, 15, 20)), {'original_axcodes': 'ALS'}, (2, 15, 10, 20)],
    [{'axcodes': 'AL'},
     np.ones((2, 10, 15)), {'original_axcodes': 'AR'}, (2, 10, 15)],
    [{'axcodes': 'L'},
     np.ones((2, 10)), {'original_axcodes': 'R'}, (2, 10)],
]


class TestOrientationCase(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_ornt(self, init_param, img, data_param, expected_shape):
        res = Orientation(**init_param)(img, **data_param)
        np.testing.assert_allclose(res[0].shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
