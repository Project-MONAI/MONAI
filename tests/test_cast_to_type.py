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

from monai.transforms import CastToType

TEST_CASE_1 = [{"dtype": np.float64}, np.array([[0, 1], [1, 2]], dtype=np.float32), np.float64]

TEST_CASE_2 = [{"dtype": torch.float64}, torch.tensor([[0, 1], [1, 2]], dtype=torch.float32), torch.float64]


class TestCastToType(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type(self, input_param, input_data, expected_type):
        result = CastToType(**input_param)(input_data)
        self.assertEqual(result.dtype, expected_type)


if __name__ == "__main__":
    unittest.main()
