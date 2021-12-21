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
import torch
from parameterized import parameterized

from monai.transforms import CenterSpatialCrop

TEST_CASE_0 = [{"roi_size": [2, 2, -1]}, np.random.randint(0, 2, size=[3, 3, 3, 3]), (3, 2, 2, 3)]

TEST_CASE_1 = [{"roi_size": [2, 2, 2]}, np.random.randint(0, 2, size=[3, 3, 3, 3]), (3, 2, 2, 2)]

TEST_CASE_2 = [
    {"roi_size": [2, 2]},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
    np.array([[[1, 2], [2, 3]]]),
]

TEST_CASE_3 = [
    {"roi_size": [2, 2, 2]},
    torch.randint(0, 2, size=[3, 3, 3, 3], device="cuda" if torch.cuda.is_available() else "cpu"),
    (3, 2, 2, 2),
]


class TestCenterSpatialCrop(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_3])
    def test_shape(self, input_param, input_data, expected_shape):
        result = CenterSpatialCrop(**input_param)(input_data)
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(input_data, torch.Tensor))
        np.testing.assert_allclose(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_2])
    def test_value(self, input_param, input_data, expected_value):
        result = CenterSpatialCrop(**input_param)(input_data)
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(input_data, torch.Tensor))
        np.testing.assert_allclose(result, expected_value)


if __name__ == "__main__":
    unittest.main()
