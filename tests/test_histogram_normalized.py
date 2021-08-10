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

import numpy as np
from parameterized import parameterized

from monai.transforms import HistogramNormalized

TEST_CASE_1 = [
    {"keys": "img", "bins": 4, "max": 4},
    {"img": np.array([0, 1, 2, 3, 4])},
    np.array([0.8, 1.6, 2.4, 4.0, 4.0]),
]

TEST_CASE_2 = [
    {"keys": "img", "bins": 4, "max": 4, "dtype": np.uint8},
    {"img": np.array([0, 1, 2, 3, 4])},
    np.array([0, 1, 2, 4, 4]),
]

TEST_CASE_3 = [
    {"keys": "img", "bins": 256, "max": 255, "dtype": np.uint8},
    {"img": np.array([[[100, 200], [150, 250]]])},
    np.array([[[63, 191], [127, 255]]]),
]


class TestHistogramNormalized(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, argments, image, expected_data):
        result = HistogramNormalized(**argments)(image)["img"]
        np.testing.assert_allclose(result, expected_data)
        self.assertEqual(result.dtype, argments.get("dtype", np.float32))


if __name__ == "__main__":
    unittest.main()
