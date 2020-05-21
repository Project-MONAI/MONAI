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

from monai.transforms import RandRotate
from tests.utils import NumpyImageTestCase2D


class TestRandRotate(NumpyImageTestCase2D):
    @parameterized.expand(
        [
            (90, (0, 1), True, 1, "reflect", 0, True),
            ((-45, 45), (1, 0), True, 3, "constant", 0, True),
            (180, (1, 0), False, 2, "constant", 4, False),
        ]
    )
    def test_correct_results(self, degrees, spatial_axes, reshape, order, mode, cval, prefilter):
        rotate_fn = RandRotate(
            degrees,
            prob=1.0,
            spatial_axes=spatial_axes,
            reshape=reshape,
            interp_order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )
        rotate_fn.set_random_state(243)
        rotated = rotate_fn(self.imt[0])

        angle = rotate_fn.angle
        expected = list()
        for channel in self.imt[0]:
            expected.append(
                scipy.ndimage.rotate(
                    channel, angle, spatial_axes, reshape, order=order, mode=mode, cval=cval, prefilter=prefilter
                )
            )
        expected = np.stack(expected).astype(np.float32)
        self.assertTrue(np.allclose(expected, rotated))


if __name__ == "__main__":
    unittest.main()
