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
from typing import List, Tuple

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data.utils import zoom_affine
from tests.utils import TEST_NDARRAYS

VALID_CASES: List[Tuple] = []
DIAGONAL_CASES: List[Tuple] = []
for p in TEST_NDARRAYS:
    VALID_CASES.append(
        (
            p,
            np.array([[2, 1, 4], [-1, -3, 5], [0, 0, 1]]),
            (10, 20, 30),
            np.array([[8.94427191, -8.94427191, 0], [-4.47213595, -17.88854382, 0], [0.0, 0.0, 1.0]]),
        )
    )
    VALID_CASES.append(
        (
            p,
            np.array([[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]]),
            (10, 20, 30),
            np.array([[10, 0, 0, 0], [0, 20, 0, 0], [0, 0, 30, 0], [0, 0, 0, 1]]),
        )
    )
    VALID_CASES.append(
        (
            p,
            np.array([[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]]),
            (10, 20),
            np.array([[10, 0, 0, 0], [0, 20, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]),
        )
    )
    VALID_CASES.append(
        (
            p,
            np.array([[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]]),
            (10,),
            np.array([[10, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]),
        )
    )
    VALID_CASES.append(
        (
            p,
            [[1, 0, 10], [0, 1, 20], [0, 0, 1]]
            @ ([[0, -1, 0], [1, 0, 0], [0, 0, 1]] @ np.array([[2, 0.3, 0], [0, 3, 0], [0, 0, 1]])),
            (4, 5, 6),
            ([[0, -1, 0], [1, 0, 0], [0, 0, 1]] @ np.array([[4, 0, 0], [0, 5, 0], [0, 0, 1]])),
        )
    )

    DIAGONAL_CASES.append(
        (
            p,
            np.array([[-1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]]),
            (10, 20, 30),
            np.array([[10, 0, 0, 0], [0, 20, 0, 0], [0, 0, 30, 0], [0, 0, 0, 1]]),
        )
    )
    DIAGONAL_CASES.append(
        (
            p,
            np.array([[2, 1, 4], [-1, -3, 5], [0, 0, 1]]),
            (10, 20, 30),
            np.array([[10, 0, 0], [0, 20, 0], [0.0, 0.0, 1.0]]),
        )
    )
    DIAGONAL_CASES.append(
        (  # test default scale from affine
            p,
            np.array([[2, 1, 4], [-1, -3, 5], [0, 0, 1]]),
            (10,),
            np.array([[10, 0, 0], [0, 3.162278, 0], [0.0, 0.0, 1.0]]),
        )
    )


class TestZoomAffine(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct(self, in_type, affine, scale, expected):
        _affine = in_type(affine)
        output = zoom_affine(_affine, scale, diagonal=False)
        ornt_affine = nib.orientations.ornt2axcodes(nib.orientations.io_orientation(output))
        ornt_output = nib.orientations.ornt2axcodes(nib.orientations.io_orientation(affine))
        np.testing.assert_array_equal(ornt_affine, ornt_output)
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)

    @parameterized.expand(DIAGONAL_CASES)
    def test_diagonal(self, in_type, affine, scale, expected):
        output = zoom_affine(in_type(affine), scale, diagonal=True)
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
