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
from monai.transforms import RandSpatialCropSamples

TEST_CASE_1 = [
    {"roi_size": [3, 3, 3], "num_samples": 4, "random_center": True},
    np.random.randint(0, 2, size=[3, 3, 3, 3]),
    (3, 3, 3, 3),
]

TEST_CASE_2 = [
    {"roi_size": [3, 3, 3], "num_samples": 8, "random_center": False},
    np.random.randint(0, 2, size=[3, 3, 3, 3]),
    (3, 3, 3, 3),
]


class TestRandSpatialCropSamples(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_shape):
        result = RandSpatialCropSamples(**input_param)(input_data)
        for item in result:
            self.assertTupleEqual(item.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
