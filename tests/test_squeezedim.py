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

from monai.transforms import SqueezeDim

TEST_CASE_1 = [{"dim": None}, np.random.rand(1, 2, 1, 3), (2, 3)]

TEST_CASE_2 = [{"dim": 2}, np.random.rand(1, 2, 1, 8, 16), (1, 2, 8, 16)]

TEST_CASE_3 = [{"dim": -1}, np.random.rand(1, 1, 16, 8, 1), (1, 1, 16, 8)]

TEST_CASE_4 = [{}, np.random.rand(1, 2, 1, 3), (2, 1, 3)]

TEST_CASE_4_PT = [{}, torch.rand(1, 2, 1, 3), (2, 1, 3)]

TEST_CASE_5 = [ValueError, {"dim": -2}, np.random.rand(1, 1, 16, 8, 1)]

TEST_CASE_6 = [TypeError, {"dim": 0.5}, np.random.rand(1, 1, 16, 8, 1)]


class TestSqueezeDim(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_4_PT])
    def test_shape(self, input_param, test_data, expected_shape):
        result = SqueezeDim(**input_param)(test_data)
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_5, TEST_CASE_6])
    def test_invalid_inputs(self, exception, input_param, test_data):
        with self.assertRaises(exception):
            SqueezeDim(**input_param)(test_data)


if __name__ == "__main__":
    unittest.main()
