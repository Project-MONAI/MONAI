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

from monai.transforms import RandGaussianSharpend

TEST_CASE_1 = [
    {"keys": "img", "prob": 1.0},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[4.754159, 4.736094, 4.754159], [10.598042, 11.24803, 10.598039], [14.249546, 16.14466, 14.249547]],
            [[19.00694, 20.396658, 19.00694], [26.495098, 28.120085, 26.49509], [28.502321, 31.805233, 28.502329]],
        ]
    ),
]

TEST_CASE_2 = [
    {
        "keys": "img",
        "sigma1_x": (0.5, 0.75),
        "sigma1_y": (0.5, 0.75),
        "sigma1_z": (0.5, 0.75),
        "sigma2_x": 0.4,
        "sigma2_y": 0.4,
        "sigma2_z": 0.4,
        "prob": 1.0,
    },
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[3.4868715, 2.693231, 3.4868698], [8.438889, 7.384708, 8.438892], [12.872246, 12.808499, 12.872242]],
            [[15.7562065, 14.319538, 15.7562065], [21.09723, 18.461775, 21.097229], [25.14158, 24.434803, 25.14158]],
        ]
    ),
]

TEST_CASE_3 = [
    {
        "keys": "img",
        "sigma1_x": (0.5, 0.75),
        "sigma1_y": (0.5, 0.75),
        "sigma1_z": (0.5, 0.75),
        "sigma2_x": (0.5, 0.75),
        "sigma2_y": (0.5, 0.75),
        "sigma2_z": (0.5, 0.75),
        "prob": 1.0,
    },
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[4.4568377, 3.4987352, 4.4568377], [11.090087, 10.003474, 11.090087], [17.025122, 17.420639, 17.025122]],
            [[20.568314, 19.188267, 20.568314], [27.725227, 25.008686, 27.725227], [33.136593, 33.11017, 33.1366]],
        ]
    ),
]


class TestRandGaussianSharpend(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, argments, image, expected_data):
        converter = RandGaussianSharpend(**argments)
        converter.set_random_state(seed=0)
        result = converter(image)
        np.testing.assert_allclose(result["img"], expected_data, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
