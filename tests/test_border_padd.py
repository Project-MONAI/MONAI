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

import numpy as np
from parameterized import parameterized

from monai.transforms import BorderPadd
from monai.utils import NumpyPadMode

TEST_CASE_1 = [
    {"keys": ["img", "seg"], "spatial_border": 2, "mode": ["constant", "edge"]},
    {"img": np.zeros((3, 8, 8, 4)), "seg": np.zeros((3, 8, 8, 4))},
    np.zeros((3, 12, 12, 8)),
]

TEST_CASE_2 = [
    {"keys": "img", "spatial_border": [1, 2, 3], "mode": "constant"},
    {"img": np.zeros((3, 8, 8, 4))},
    np.zeros((3, 10, 12, 10)),
]

TEST_CASE_3 = [
    {"keys": "img", "spatial_border": [1, 2, 3, 4, 5, 6], "mode": "constant"},
    {"img": np.zeros((3, 8, 8, 4))},
    np.zeros((3, 11, 15, 15)),
]

TEST_CASE_4 = [
    {"keys": ["img", "seg"], "spatial_border": 2, "mode": ["constant", NumpyPadMode.EDGE]},
    {"img": np.zeros((3, 8, 8, 4)), "seg": np.zeros((3, 8, 8, 4))},
    np.zeros((3, 12, 12, 8)),
]

TEST_CASE_5 = [
    {"keys": ["img", "seg"], "spatial_border": 2, "mode": [NumpyPadMode.CONSTANT, NumpyPadMode.EDGE]},
    {"img": np.zeros((3, 8, 8, 4)), "seg": np.zeros((3, 8, 8, 4))},
    np.zeros((3, 12, 12, 8)),
]


class TestBorderPadd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_pad_shape(self, input_param, input_data, expected_val):
        padder = BorderPadd(**input_param)
        result = padder(input_data)
        self.assertAlmostEqual(result["img"].shape, expected_val.shape)


if __name__ == "__main__":
    unittest.main()
