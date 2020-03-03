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

from monai.transforms import Flip
from tests.utils import NumpyImageTestCase2D


class FlipTest(NumpyImageTestCase2D):

    @parameterized.expand([
        ("no_axis", None, AssertionError),
        ("wrong_axis", ['s', 1], AssertionError)
    ])
    def test_invalid_inputs(self, _, axis, raises):
        with self.assertRaises(raises):
            flip = Flip(axis)
            flip(self.imt)

    def test_correct_results(self):
        flip = Flip(axes=[0, 1])
        expected = np.flip(self.imt, [0, 1])
        self.assertTrue(np.allclose(expected, flip(self.imt)))


if __name__ == '__main__':
    unittest.main()
