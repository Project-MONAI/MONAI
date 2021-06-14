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
from typing import List, Tuple

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms.transform import convert_data_type
from monai.utils.misc import dtype_convert

TESTS: List[Tuple] = []
TESTS.append((np.array, torch.Tensor, np.float32, torch.float32))
TESTS.append((torch.Tensor, np.ndarray, np.float32, torch.float32))
TESTS.append((np.array, torch.Tensor, torch.float32, np.float32))
TESTS.append((torch.Tensor, np.ndarray, torch.float32, np.float32))


class TestConvertDataType(unittest.TestCase):
    @staticmethod
    def get_im(im_type, dtype):
        dtype = dtype_convert(dtype, im_type)
        lib = torch if im_type is torch.Tensor else np
        return lib.zeros((1, 2, 3), dtype=dtype)

    @parameterized.expand(TESTS)
    def test_convert_data_type(self, in_type, out_type, in_dtype, out_dtype):
        orig_im = self.get_im(in_type, in_dtype)
        converted_im, orig_type, _ = convert_data_type(orig_im, out_type)
        self.assertEqual(type(orig_im), orig_type)
        self.assertEqual(type(converted_im), out_type)


if __name__ == "__main__":
    unittest.main()
