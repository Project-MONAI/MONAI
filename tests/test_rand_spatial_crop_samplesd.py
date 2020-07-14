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
from monai.transforms import RandSpatialCropSamplesd

TEST_CASE_1 = [
    {"keys": ["img", "seg"], "num_samples": 4, "roi_size": [3, 3, 3], "random_center": True},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3]), "seg": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 3, 3, 3),
]

TEST_CASE_2 = [
    {"keys": ["img", "seg"], "num_samples": 8, "roi_size": [3, 3, 3], "random_center": False},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3]), "seg": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 3, 3, 3),
]


class TestRandSpatialCropSamplesd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_shape):
        result = RandSpatialCropSamplesd(**input_param)(input_data)
        for item in result:
            self.assertTupleEqual(item["img"].shape, expected_shape)
            self.assertTupleEqual(item["seg"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
