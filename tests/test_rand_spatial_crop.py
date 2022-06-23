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

from monai.transforms import RandSpatialCrop
from tests.utils import TEST_NDARRAYS_ALL, assert_allclose

TEST_CASE_0 = [
    {"roi_size": [3, 3, -1], "random_center": True},
    np.random.randint(0, 2, size=[3, 3, 3, 4]),
    (3, 3, 3, 4),
]

TEST_CASE_1 = [{"roi_size": [3, 3, 3], "random_center": True}, np.random.randint(0, 2, size=[3, 3, 3, 3]), (3, 3, 3, 3)]

TEST_CASE_2 = [
    {"roi_size": [3, 3, 3], "random_center": False},
    np.random.randint(0, 2, size=[3, 3, 3, 3]),
    (3, 3, 3, 3),
]

TEST_CASE_3 = [
    {"roi_size": [3, 3], "random_center": False},
    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
]

TEST_CASE_4 = [
    {"roi_size": [3, 3, 3], "max_roi_size": [5, -1, 4], "random_center": True, "random_size": True},
    np.random.randint(0, 2, size=[1, 4, 5, 6]),
    (1, 4, 4, 3),
]

TEST_CASE_5 = [
    {"roi_size": 3, "max_roi_size": 4, "random_center": True, "random_size": True},
    np.random.randint(0, 2, size=[1, 4, 5, 6]),
    (1, 3, 4, 3),
]


class TestRandSpatialCrop(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_shape):
        result = RandSpatialCrop(**input_param)(input_data)
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_3])
    def test_value(self, input_param, input_data):
        for p in TEST_NDARRAYS_ALL:
            cropper = RandSpatialCrop(**input_param)
            result = cropper(p(input_data))
            roi = [(2 - i // 2, 2 + i - i // 2) for i in cropper._size]
            assert_allclose(result, input_data[:, roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]], type_test=False)

    @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
    def test_random_shape(self, input_param, input_data, expected_shape):
        cropper = RandSpatialCrop(**input_param)
        cropper.set_random_state(seed=123)
        result = cropper(input_data)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
