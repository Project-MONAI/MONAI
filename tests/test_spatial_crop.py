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

from monai.transforms import SpatialCrop

TEST_CASES = [
    [
        {"roi_center": [1, 1, 1], "roi_size": [2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
    ],
    [{"roi_start": [0, 0, 0], "roi_end": [2, 2, 2]}, (3, 3, 3, 3), (3, 2, 2, 2)],
    [{"roi_start": [0, 0], "roi_end": [2, 2]}, (3, 3, 3, 3), (3, 2, 2, 3)],
    [
        {"roi_start": [0, 0, 0, 0, 0], "roi_end": [2, 2, 2, 2, 2]},
        (3, 3, 3, 3),
        (3, 2, 2, 2),
    ],
    [
        {"roi_start": [0, 0, 0, 0, 0], "roi_end": [8, 8, 8, 2, 2]},
        (3, 3, 3, 3),
        (3, 3, 3, 3),
    ],
    [
        {"roi_start": [1, 0, 0], "roi_end": [1, 8, 8]},
        (3, 3, 3, 3),
        (3, 0, 3, 3),
    ],
]


class TestSpatialCrop(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        input_data = np.random.randint(0, 2, size=input_shape)
        result = SpatialCrop(**input_param)(input_data)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
