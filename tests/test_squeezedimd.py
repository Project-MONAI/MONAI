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
import torch
from parameterized import parameterized

from monai.transforms import SqueezeDimd

TEST_CASE_1 = [
    {"keys": ["img", "seg"], "dim": None},
    {"img": np.random.rand(1, 2, 1, 3), "seg": np.random.randint(0, 2, size=[1, 2, 1, 3])},
    (2, 3),
]

TEST_CASE_2 = [
    {"keys": ["img", "seg"], "dim": 2},
    {"img": np.random.rand(1, 2, 1, 8, 16), "seg": np.random.randint(0, 2, size=[1, 2, 1, 8, 16])},
    (1, 2, 8, 16),
]

TEST_CASE_3 = [
    {"keys": ["img", "seg"], "dim": -1},
    {"img": np.random.rand(1, 1, 16, 8, 1), "seg": np.random.randint(0, 2, size=[1, 1, 16, 8, 1])},
    (1, 1, 16, 8),
]

TEST_CASE_4 = [
    {"keys": ["img", "seg"]},
    {"img": np.random.rand(1, 2, 1, 3), "seg": np.random.randint(0, 2, size=[1, 2, 1, 3])},
    (2, 1, 3),
]

TEST_CASE_4_PT = [
    {"keys": ["img", "seg"], "dim": 0},
    {"img": torch.rand(1, 2, 1, 3), "seg": torch.randint(0, 2, size=[1, 2, 1, 3])},
    (2, 1, 3),
]

TEST_CASE_5 = [
    ValueError,
    {"keys": ["img", "seg"], "dim": -2},
    {"img": np.random.rand(1, 1, 16, 8, 1), "seg": np.random.randint(0, 2, size=[1, 1, 16, 8, 1])},
]

TEST_CASE_6 = [
    TypeError,
    {"keys": ["img", "seg"], "dim": 0.5},
    {"img": np.random.rand(1, 1, 16, 8, 1), "seg": np.random.randint(0, 2, size=[1, 1, 16, 8, 1])},
]


class TestSqueezeDim(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_4_PT])
    def test_shape(self, input_param, test_data, expected_shape):
        result = SqueezeDimd(**input_param)(test_data)
        self.assertTupleEqual(result["img"].shape, expected_shape)
        self.assertTupleEqual(result["seg"].shape, expected_shape)

    @parameterized.expand([TEST_CASE_5, TEST_CASE_6])
    def test_invalid_inputs(self, exception, input_param, test_data):
        with self.assertRaises(exception):
            SqueezeDimd(**input_param)(test_data)


if __name__ == "__main__":
    unittest.main()
