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

from monai.transforms import CenterScaleCropd

TEST_CASE_0 = [{"keys": "img", "roi_scale": [0.6, 0.3, -1]}, np.random.randint(0, 2, size=[3, 3, 3, 3]), (3, 2, 1, 3)]

TEST_CASE_1 = [{"keys": "img", "roi_scale": 0.6}, np.random.randint(0, 2, size=[3, 3, 3, 3]), (3, 2, 2, 2)]

TEST_CASE_2 = [
    {"keys": "img", "roi_scale": [0.4, 0.4]},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
    np.array([[[1, 2], [2, 3]]]),
]

TEST_CASE_3 = [
    {"keys": "img", "roi_scale": 0.5},
    torch.randint(0, 2, size=[3, 3, 3, 3], device="cuda" if torch.cuda.is_available() else "cpu"),
    (3, 2, 2, 2),
]

TEST_CASE_4 = [
    {"keys": "test", "roi_scale": 0.6, "allow_missing_keys": True},
    np.random.randint(0, 2, size=[3, 3, 3, 3]),
    (3, 3, 3, 3),
]


class TestCenterScaleCropd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_3, TEST_CASE_4])
    def test_shape(self, input_param, input_data, expected_shape):
        result = CenterScaleCropd(**input_param)({"img": input_data})
        np.testing.assert_allclose(result["img"].shape, expected_shape)

    @parameterized.expand([TEST_CASE_2])
    def test_value(self, input_param, input_data, expected_value):
        result = CenterScaleCropd(**input_param)({"img": input_data})
        np.testing.assert_allclose(result["img"], expected_value)


if __name__ == "__main__":
    unittest.main()
