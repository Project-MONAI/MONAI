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

from monai.metrics import compute_average_surface_distance, compute_percent_hausdorff_distance, get_surface_distance


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
        # following list contains results of [HD, 95th HD, assd, asd]
        [0, 0, 0, 0],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20)),
            create_spherical_seg_3d(radius=20, centre=(19, 19, 19)),
            1,
        ],
        # following list contains results of [HD, 95th HD, assd, asd]
        [1.7320508075688772, 1.7320508075688772, 0.8957429366433527, 0.8958613558852289],
    ],
    [
        [
            create_spherical_seg_3d(radius=33, labelfield_value=2, centre=(19, 33, 22)),
            create_spherical_seg_3d(radius=33, labelfield_value=2, centre=(20, 33, 22)),
            2,
        ],
        # following list contains results of [HD, 95th HD, assd, asd]
        [1, 1, 0.35021200688332677, 0.3483278807706289],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            1,
        ],
        # following list contains results of [HD, 95th HD, assd, asd]
        [20.223748416156685, 20.049937655763422, 13.975673696300824, 12.040033513150455],
    ],
]


class TestAllSurfaceMetrics(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        [seg_1, seg_2, label_idx] = input_data
        [hd, hd_95, assd, asd] = expected_value
        # test both with and without crop methods
        for crop in [True, False]:
            suf1 = get_surface_distance(seg_1, seg_2, label_idx, crop)
            suf2 = get_surface_distance(seg_2, seg_1, label_idx, crop)
            result_hd = compute_percent_hausdorff_distance(suf1, suf2)
            result_hd_95 = compute_percent_hausdorff_distance(suf1, suf2, percent=95)
            # test maximum Hausdorff Distance
            np.testing.assert_allclose(hd, result_hd, rtol=1e-7)
            # test 95th percentile of the Hausdorff Distance
            np.testing.assert_allclose(hd_95, result_hd_95, rtol=1e-7)

        result_assd = compute_average_surface_distance(seg_1, seg_2, label_idx, symmetric=True)
        result_asd = compute_average_surface_distance(seg_1, seg_2, label_idx, symmetric=False)
        # test Average Symmetric Surface Distance
        np.testing.assert_allclose(assd, result_assd, rtol=1e-7)
        # test Average Surface Distance
        np.testing.assert_allclose(asd, result_asd, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
