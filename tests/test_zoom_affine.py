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

from monai.data.utils import zoom_affine

VALID_CASES = [
    (
        np.array([[2, 1, 4],
                  [-1, -3, 5],
                  [0, 0, 1]],),
        (10, 20, 30),
        np.array([[  8.94427191,  -8.94427191,  17.88854382],
                  [ -4.47213595, -17.88854382,  44.72135955],
                  [  0.        ,   0.        ,   1.        ]],),
    ),
    (
        np.array([[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]],),
        (10, 20, 30),
        np.array([[10, 0, 0, 40], [0, 20, 0, 50], [0, 0, 30, 60], [0, 0, 0, 1]],),
    ),
    (
        np.array([[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]],),
        (10, 20),
        np.array([[10, 0, 0, 40], [0, 20, 0, 50], [0, 0, 3, 6], [0, 0, 0, 1]],),
    ),
    (
        np.array([[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]],),
        (10,),
        np.array([[10, 0, 0, 40], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]],),
    ),
    (
        [[1, 0, 10], [0, 1, 20], [0, 0, 1]] @
        ([[0, -1, 0], [1, 0, 0], [0, 0, 1]] @
         np.array([[2, 0.3, 0], [0, 3, 0], [0, 0, 1]])),
        (4, 5, 6),
        [[1, 0, 20], [0, 1, 33.3333333], [0, 0, 1]] @
        ([[0, -1, 0], [1, 0, 0], [0, 0, 1]] @
         np.array([[4, 0, 0],
                   [0, 5, 0],
                   [0, 0, 1]])),
    ),
]

DIAGONAL_CASES = [
    (
        np.array([[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]],),
        (10, 20, 30),
        np.array([[10, 0, 0, 0], [0, 20, 0, 0], [0, 0, 30, 0], [0, 0, 0, 1]],),
    ),
]


class TestZoomAffine(unittest.TestCase):

    @parameterized.expand(VALID_CASES)
    def test_correct(self, affine, scale, expected):
        output = zoom_affine(affine, scale, diagonal=False)
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)

    @parameterized.expand(DIAGONAL_CASES)
    def test_diagonal(self, affine, scale, expected):
        output = zoom_affine(affine, scale)
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)



if __name__ == '__main__':
    unittest.main()
