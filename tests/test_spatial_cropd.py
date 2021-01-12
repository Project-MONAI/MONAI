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

from monai.transforms import SpatialCropd

TEST_CASE_1 = [
    {"keys": ["img"], "roi_center": [1, 1, 1], "roi_size": [2, 2, 2]},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 2, 2, 2),
]

TEST_CASE_2 = [
    {"keys": ["img"], "roi_start": [0, 0, 0], "roi_end": [2, 2, 2]},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 2, 2, 2),
]

TEST_CASE_3 = [
    {"keys": ["img"], "roi_start": [0, 0], "roi_end": [2, 2]},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 2, 2, 3),
]

TEST_CASE_4 = [
    {"keys": ["img"], "roi_start": [0, 0, 0, 0, 0], "roi_end": [2, 2, 2, 2, 2]},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 2, 2, 2),
]


class TestSpatialCropd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_shape(self, input_param, input_data, expected_shape):
        result = SpatialCropd(**input_param)(input_data)
        self.assertTupleEqual(result["img"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
