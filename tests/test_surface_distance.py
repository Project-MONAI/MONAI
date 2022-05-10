# Copyright (c) MONAI Consortium
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
import torch
from parameterized import parameterized

from monai.metrics import SurfaceDistanceMetric


def create_spherical_seg_3d(
    radius: float = 20.0, centre: Tuple[int, int, int] = (49, 49, 49), im_shape: Tuple[int, int, int] = (99, 99, 99)
) -> np.ndarray:
    """
    Return a 3D image with a sphere inside. Voxel values will be
    1 inside the sphere, and 0 elsewhere.

    Args:
        radius: radius of sphere (in terms of number of voxels, can be partial)
        centre: location of sphere centre.
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

    image[circle] = 1
    image[~circle] = 0
    return image


TEST_CASES = [
    [[create_spherical_seg_3d(), create_spherical_seg_3d()], [0, 0]],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20)),
            create_spherical_seg_3d(radius=20, centre=(19, 19, 19)),
            "taxicab",
        ],
        [1.0380029806259314, 1.0380029806259314],
    ],
    [
        [
            create_spherical_seg_3d(radius=33, centre=(19, 33, 22)),
            create_spherical_seg_3d(radius=33, centre=(20, 33, 22)),
        ],
        [0.350217, 0.3483278807706289],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
        ],
        [15.117741, 12.040033513150455],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            "chessboard",
        ],
        [11.492719, 9.605067064083457],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            "taxicab",
        ],
        [20.214613, 12.432687531048186],
    ],
    [[np.zeros([99, 99, 99]), create_spherical_seg_3d(radius=40, centre=(20, 33, 22))], [np.inf, np.inf]],
    [[create_spherical_seg_3d(), np.zeros([99, 99, 99]), "taxicab"], [np.inf, np.inf]],
]

TEST_CASES_NANS = [
    [
        [
            # both pred and gt do not have foreground, metric and not_nans should be 0
            np.zeros([99, 99, 99]),
            np.zeros([99, 99, 99]),
        ]
    ]
]


class TestAllSurfaceMetrics(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        if len(input_data) == 3:
            [seg_1, seg_2, metric] = input_data
        else:
            [seg_1, seg_2] = input_data
            metric = "euclidean"
        ct = 0
        seg_1 = torch.tensor(seg_1)
        seg_2 = torch.tensor(seg_2)
        for symmetric in [True, False]:
            sur_metric = SurfaceDistanceMetric(include_background=False, symmetric=symmetric, distance_metric=metric)
            # shape of seg_1, seg_2 are: HWD, converts to BNHWD
            batch, n_class = 2, 3
            batch_seg_1 = seg_1.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
            batch_seg_2 = seg_2.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
            sur_metric(batch_seg_1, batch_seg_2)
            result = sur_metric.aggregate()
            expected_value_curr = expected_value[ct]
            np.testing.assert_allclose(expected_value_curr, result, rtol=1e-5)
            ct += 1

    @parameterized.expand(TEST_CASES_NANS)
    def test_nans(self, input_data):
        [seg_1, seg_2] = input_data
        seg_1 = torch.tensor(seg_1)
        seg_2 = torch.tensor(seg_2)
        sur_metric = SurfaceDistanceMetric(include_background=False, get_not_nans=True)
        # test list of channel-first Tensor
        batch_seg_1 = [seg_1.unsqueeze(0)]
        batch_seg_2 = [seg_2.unsqueeze(0)]
        sur_metric(batch_seg_1, batch_seg_2)
        result, not_nans = sur_metric.aggregate(reduction="mean")
        np.testing.assert_allclose(0, result, rtol=1e-5)
        np.testing.assert_allclose(0, not_nans, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
