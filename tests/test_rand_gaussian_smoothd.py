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

from monai.transforms import RandGaussianSmoothd

TEST_CASE_1 = [
    {"keys": "img", "sigma_x": (0.5, 1.5), "prob": 1.0},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[0.7291442, 0.9260285, 0.7291442], [1.1054044, 1.4038869, 1.1054044], [1.0672514, 1.3554319, 1.0672514]],
            [[2.076441, 2.6371238, 2.076441], [2.763511, 3.5097172, 2.763511], [2.4145484, 3.0665274, 2.4145484]],
        ]
    ),
]

TEST_CASE_2 = [
    {"keys": "img", "sigma_x": (0.5, 1.5), "sigma_y": (0.5, 1.0), "prob": 1.0},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[0.78625894, 1.0031066, 0.78625894], [1.1919919, 1.5207394, 1.191992], [1.1508504, 1.4682512, 1.1508505]],
            [[2.239091, 2.8566248, 2.239091], [2.97998, 3.8018486, 2.97998], [2.6036828, 3.3217697, 2.6036828]],
        ]
    ),
]


class TestRandGaussianSmoothd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_value(self, argments, image, expected_data):
        converter = RandGaussianSmoothd(**argments)
        converter.set_random_state(seed=0)
        result = converter(image)
        np.testing.assert_allclose(result["img"], expected_data, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
