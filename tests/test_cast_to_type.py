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
from monai.utils.type_conversion import get_equivalent_dtype
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    for out_dtype in (np.float64, torch.float64):
        TESTS.append([out_dtype, p(np.array([[0, 1], [1, 2]], dtype=np.float32)), out_dtype])


class TestCastToType(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_type(self, out_dtype, input_data, expected_type):

        result = CastToType(dtype=out_dtype)(input_data)
        self.assertEqual(result.dtype, get_equivalent_dtype(expected_type, type(result)))

        result = CastToType()(input_data, out_dtype)
        self.assertEqual(result.dtype, get_equivalent_dtype(expected_type, type(result)))


if __name__ == "__main__":
    unittest.main()
