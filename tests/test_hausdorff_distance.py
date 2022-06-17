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

from monai.metrics import HausdorffDistanceMetric


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
    [[create_spherical_seg_3d(), create_spherical_seg_3d(), 1], [0, 0, 0, 0, 0, 0]],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20)),
            create_spherical_seg_3d(radius=20, centre=(19, 19, 19)),
        ],
        [1.7320508075688772, 1.7320508075688772, 1, 1, 3, 3],
    ],
    [
        [
            create_spherical_seg_3d(radius=33, centre=(19, 33, 22)),
            create_spherical_seg_3d(radius=33, centre=(20, 33, 22)),
        ],
        [1, 1, 1, 1, 1, 1],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
        ],
        [20.09975124224178, 20.223748416156685, 15, 20, 24, 35],
    ],
    [
        [
            # pred does not have foreground (but gt has), the metric should be inf
            np.zeros([99, 99, 99]),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
        ],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    ],
    [
        [
            # gt does not have foreground (but pred has), the metric should be inf
            create_spherical_seg_3d(),
            np.zeros([99, 99, 99]),
        ],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            95,
        ],
        [19.924858845171276, 20.09975124224178, 14, 18, 22, 33],
    ],
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


class TestHausdorffDistance(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        percentile = None
        if len(input_data) == 3:
            [seg_1, seg_2, percentile] = input_data
        else:
            [seg_1, seg_2] = input_data
        ct = 0
        seg_1 = torch.tensor(seg_1)
        seg_2 = torch.tensor(seg_2)
        for metric in ["euclidean", "chessboard", "taxicab"]:
            for directed in [True, False]:
                hd_metric = HausdorffDistanceMetric(
                    include_background=False, distance_metric=metric, percentile=percentile, directed=directed
                )
                # shape of seg_1, seg_2 are: HWD, converts to BNHWD
                batch, n_class = 2, 3
                batch_seg_1 = seg_1.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
                batch_seg_2 = seg_2.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
                hd_metric(batch_seg_1, batch_seg_2)
                result = hd_metric.aggregate(reduction="mean")
                expected_value_curr = expected_value[ct]
                np.testing.assert_allclose(expected_value_curr, result, rtol=1e-7)
                ct += 1

    @parameterized.expand(TEST_CASES_NANS)
    def test_nans(self, input_data):
        [seg_1, seg_2] = input_data
        seg_1 = torch.tensor(seg_1)
        seg_2 = torch.tensor(seg_2)
        hd_metric = HausdorffDistanceMetric(include_background=False, get_not_nans=True)
        batch_seg_1 = seg_1.unsqueeze(0).unsqueeze(0)
        batch_seg_2 = seg_2.unsqueeze(0).unsqueeze(0)
        hd_metric(batch_seg_1, batch_seg_2)
        result, not_nans = hd_metric.aggregate()
        np.testing.assert_allclose(0, result, rtol=1e-7)
        np.testing.assert_allclose(0, not_nans, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
