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

from monai.metrics import compute_average_surface_distance


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


TEST_CASES = [
    [
        [create_spherical_seg_3d(), create_spherical_seg_3d(), 1],
        [0, 0],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20)),
            create_spherical_seg_3d(radius=20, centre=(19, 19, 19)),
            1,
            "taxicab",
        ],
        [1.0380029806259314, 1.0380029806259314],
    ],
    [
        [
            create_spherical_seg_3d(radius=33, labelfield_value=2, centre=(19, 33, 22)),
            create_spherical_seg_3d(radius=33, labelfield_value=2, centre=(20, 33, 22)),
            2,
        ],
        [0.35021200688332677, 0.3483278807706289],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            1,
        ],
        [13.975673696300824, 12.040033513150455],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            1,
            "chessboard",
        ],
        [10.792254295459173, 9.605067064083457],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            1,
            "taxicab",
        ],
        [17.32691760951026, 12.432687531048186],
    ],
    [
        [
            np.zeros([99, 99, 99]),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            1,
        ],
        [np.inf, np.inf],
    ],
    [
        [
            np.zeros([99, 99, 99]),
            np.zeros([99, 99, 99]),
            1,
        ],
        [np.inf, np.inf],
    ],
    [
        [
            create_spherical_seg_3d(),
            np.zeros([99, 99, 99]),
            1,
            "taxicab",
        ],
        [np.inf, np.inf],
    ],
]


class TestAllSurfaceMetrics(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        if len(input_data) == 4:
            [seg_1, seg_2, label_idx, metric] = input_data
        else:
            [seg_1, seg_2, label_idx] = input_data
            metric = "euclidean"
        ct = 0
        for symmetric in [True, False]:
            expected_value_curr = expected_value[ct]
            result = compute_average_surface_distance(
                seg_1, seg_2, label_idx, symmetric=symmetric, distance_metric=metric
            )
            np.testing.assert_allclose(expected_value_curr, result, rtol=1e-7)
            ct += 1


if __name__ == "__main__":
    unittest.main()
