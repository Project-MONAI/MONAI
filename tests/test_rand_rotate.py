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
from tests.utils import NumpyImageTestCase2D, NumpyImageTestCase3D

from monai.transforms import RandRotate


class TestRandRotate2D(NumpyImageTestCase2D):
    @parameterized.expand(
        [
            (90, True, "bilinear", "border", False),
            (45, True, "nearest", "border", False),
            (180, False, "nearest", "zeros", True),
            ((-45, 0), False, "nearest", "zeros", True),
        ]
    )
    def test_correct_results(self, degrees, keep_size, mode, padding_mode, align_corners):
        rotate_fn = RandRotate(
            range_x=degrees,
            prob=1.0,
            keep_size=keep_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        rotate_fn.set_random_state(243)
        rotated = rotate_fn(self.imt[0])

        _order = 0 if mode == "nearest" else 1
        if mode == "border":
            _mode = "nearest"
        elif mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        angle = rotate_fn.x
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -angle, (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(np.float32)
        np.testing.assert_allclose(expected, rotated[0])


class TestRandRotate3D(NumpyImageTestCase3D):
    @parameterized.expand(
        [
            (90, -30, (0.0, 180), False, "bilinear", "border", False, (1, 87, 104, 109)),
            (45, (-20, 40), (20, 30), False, "nearest", "border", True, (1, 89, 105, 104)),
            (0.0, (360, 370), (-1, 1), True, "nearest", "zeros", True, (1, 48, 64, 80)),
            ((-45, 0), 0, 0, False, "nearest", "zeros", False, (1, 48, 77, 90)),
        ]
    )
    def test_correct_results(self, x, y, z, keep_size, mode, padding_mode, align_corners, expected):
        rotate_fn = RandRotate(
            range_x=x,
            range_y=y,
            range_z=z,
            prob=1.0,
            keep_size=keep_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        rotate_fn.set_random_state(243)
        rotated = rotate_fn(self.imt[0])
        np.testing.assert_allclose(rotated.shape, expected)


if __name__ == "__main__":
    unittest.main()
