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

from monai.transforms import Rotated
from tests.utils import NumpyImageTestCase2D

TEST_CASES = [
    (90, (0, 1), True, 1, "reflect", 0, True),
    (-90, (1, 0), True, 3, "constant", 0, True),
    (180, (1, 0), False, 2, "constant", 4, False),
]


class TestRotated(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES)
    def test_correct_results(self, angle, spatial_axes, reshape, order, mode, cval, prefilter):
        key = "img"
        rotate_fn = Rotated(key, angle, spatial_axes, reshape, order, mode, cval, prefilter)
        rotated = rotate_fn({key: self.imt[0]})
        expected = list()
        for channel in self.imt[0]:
            expected.append(
                scipy.ndimage.rotate(
                    channel, angle, spatial_axes, reshape, order=order, mode=mode, cval=cval, prefilter=prefilter
                )
            )
        expected = np.stack(expected).astype(np.float32)
        self.assertTrue(np.allclose(expected, rotated[key]))


if __name__ == "__main__":
    unittest.main()
