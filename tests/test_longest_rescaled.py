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

from monai.transforms import LongestRescaled

TEST_CASE_0 = [{"keys": "img", "spatial_size": 15}, (6, 11, 15)]

TEST_CASE_1 = [{"keys": "img", "spatial_size": 15, "mode": "area"}, (6, 11, 15)]

TEST_CASE_2 = [{"keys": "img", "spatial_size": 6, "mode": "trilinear", "align_corners": True}, (3, 5, 6)]

TEST_CASE_3 = [
    {"keys": ["img", "label"], "spatial_size": 6, "mode": ["trilinear", "nearest"], "align_corners": [True, None]},
    (3, 5, 6),
]


class TestLongestRescaled(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_param, expected_shape):
        input_data = {
            "img": np.random.randint(0, 2, size=[3, 4, 7, 10]),
            "label": np.random.randint(0, 2, size=[3, 4, 7, 10]),
        }
        rescaler = LongestRescaled(**input_param)
        result = rescaler(input_data)
        for k in rescaler.keys:
            np.testing.assert_allclose(result[k].shape[1:], expected_shape)


if __name__ == "__main__":
    unittest.main()
