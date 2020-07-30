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

from monai.transforms import GaussianSharpend

TEST_CASE_1 = [
    {"keys": "img"},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[4.0335875, 3.362756, 4.0335875], [3.588128, 2.628216, 3.588128], [4.491922, 3.8134987, 4.491922]],
            [[10.427719, 8.744948, 10.427719], [8.97032, 6.5705404, 8.970321], [10.886056, 9.195692, 10.886056]],
        ]
    ),
]

TEST_CASE_2 = [
    {"keys": "img", "sigma1": 1.0, "sigma2": 0.75, "alpha": 20},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[4.146659, 4.392873, 4.146659], [8.031006, 8.804623, 8.031005], [10.127394, 11.669131, 10.127394]],
            [[14.852196, 16.439377, 14.852201], [20.077503, 22.011555, 20.077507], [20.832941, 23.715641, 20.832935]],
        ]
    ),
]

TEST_CASE_3 = [
    {"keys": "img", "sigma1": (0.5, 1.0), "sigma2": (0.5, 0.75), "alpha": 20},
    {"img": np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])},
    np.array(
        [
            [[3.129089, 3.0711129, 3.129089], [6.783306, 6.8526435, 6.7833037], [11.901203, 13.098082, 11.901203]],
            [[14.401806, 15.198004, 14.401809], [16.958261, 17.131605, 16.958261], [23.17392, 25.224974, 23.17392]],
        ]
    ),
]


class TestGaussianSharpend(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, argments, image, expected_data):
        result = GaussianSharpend(**argments)(image)
        np.testing.assert_allclose(result["img"], expected_data, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
