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

from monai.transforms import generate_pos_neg_label_crop_centers

TEST_CASE_1 = [
    {
        "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "spatial_size": [2, 2, 2],
        "num_samples": 2,
        "pos_ratio": 1.0,
        "image": None,
        "image_threshold": 0,
        "rand_state": np.random.RandomState(),
    },
    list,
    2,
    3,
]


class TestGeneratePosNegLabelCropCenters(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_type_shape(self, input_data, expected_type, expected_count, expected_shape):
        result = generate_pos_neg_label_crop_centers(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertEqual(len(result), expected_count)
        self.assertEqual(len(result[0]), expected_shape)


if __name__ == "__main__":
    unittest.main()
