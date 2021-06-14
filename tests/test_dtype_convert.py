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

from monai.utils.misc import dtype_convert

TEST_NDARRAYS = [torch.Tensor, np.ndarray]
DTYPES = [torch.float32, np.float32, np.dtype(np.float32)]

TESTS = []
for im_type in TEST_NDARRAYS:
    for im_dtype in DTYPES:
        TESTS.append((im_type, im_dtype))


class TestDtypeConvert(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_dtype_convert(self, im_type, desired_dtype):
        out = dtype_convert(desired_dtype, im_type)


if __name__ == "__main__":
    unittest.main()
