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

from monai.transforms import CastToTyped

TEST_CASE_1 = [
    {"keys": ["img"], "dtype": np.float64},
    {"img": np.array([[0, 1], [1, 2]], dtype=np.float32), "seg": np.array([[0, 1], [1, 2]], dtype=np.int8)},
    {"img": np.float64, "seg": np.int8},
]

TEST_CASE_2 = [
    {"keys": ["img"], "dtype": torch.float64},
    {
        "img": torch.tensor([[0, 1], [1, 2]], dtype=torch.float32),
        "seg": torch.tensor([[0, 1], [1, 2]], dtype=torch.int8),
    },
    {"img": torch.float64, "seg": torch.int8},
]


class TestCastToTyped(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type(self, input_param, input_data, expected_type):
        result = CastToTyped(**input_param)(input_data)
        for k, v in result.items():
            self.assertEqual(v.dtype, expected_type[k])


if __name__ == "__main__":
    unittest.main()
