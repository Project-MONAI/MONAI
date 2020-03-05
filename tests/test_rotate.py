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

import scipy.ndimage
from parameterized import parameterized

from monai.transforms import Rotate
from tests.utils import NumpyImageTestCase2D


class RotateTest(NumpyImageTestCase2D):

    @parameterized.expand([
        (90, (1, 2), True, 1, 'reflect', 0, True),
        (-90, (2, 1), True, 3, 'constant', 0, True),
        (180, (2, 3), False, 2, 'constant', 4, False),
    ])
    def test_correct_results(self, angle, axes, reshape, 
                             order, mode, cval, prefilter):
        rotate_fn = Rotate(angle, axes, reshape, 
                           order, mode, cval, prefilter)
        rotated = rotate_fn(self.imt)

        expected = scipy.ndimage.rotate(self.imt, angle, axes, reshape, order=order,
                                        mode=mode, cval=cval, prefilter=prefilter)
        self.assertTrue(np.allclose(expected, rotated))


if __name__ == '__main__':
    unittest.main()
