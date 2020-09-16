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
from typing import Tuple

import numpy as np
from parameterized import parameterized

from monai.metrics import compute_hausdorff_distance


def create_spherical_seg_3d(
    radius: float = 20.0,
    centre: Tuple[int, int, int] = (49, 49, 49),
    labelfield_value: int = 1,
    background_value: int = 0,
    im_shape: Tuple[int, int, int] = (99, 99, 99),
) -> np.ndarray:
    """
    Return a 3D image with a sphere inside. Voxel values will be
    `labelfield_value` inside the sphere, and `background_value` elsewhere.

    Args:
        radius: radius of sphere (in terms of number of voxels, can be partial)
        centre: location of sphere centre.
        labelfield_value: index of labelfield.
        background_value: index of background.
        im_shape: shape of image to create

    See also:
        :py:meth:`~create_test_image_3d`
    """
    # Create image
    image = np.zeros(im_shape, dtype=np.int32)
    spy, spx, spz = np.ogrid[
        -centre[0] : im_shape[0] - centre[0], -centre[1] : im_shape[1] - centre[1], -centre[2] : im_shape[2] - centre[2]
    ]
    circle = (spx * spx + spy * spy + spz * spz) <= radius * radius

    image[circle] = labelfield_value
    image[~circle] = background_value
    return image


# test the metric under 2 class classification task
TEST_CASES = [
    [
        {
            "seg_1": create_spherical_seg_3d(),
            "seg_2": create_spherical_seg_3d(),
            "label_idx": 1,
        },
        0.0,
    ],
    [
        {
            "seg_1": create_spherical_seg_3d(radius=20, labelfield_value=2),
            "seg_2": create_spherical_seg_3d(radius=21, labelfield_value=2),
            "label_idx": 2,
        },
        1.4142135623730951,
    ],
    [
        {
            "seg_1": create_spherical_seg_3d(radius=20, centre=(49, 49, 49)),
            "seg_2": create_spherical_seg_3d(radius=20, centre=(50, 49, 49)),
            "label_idx": 1,
        },
        1.0,
    ],
]


class TestHausdorffDistance(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        result = compute_hausdorff_distance(**input_data)
        np.testing.assert_allclose(expected_value, result, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
