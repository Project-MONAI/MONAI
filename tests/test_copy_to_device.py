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
from unittest.case import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.utils import copy_to_device

TEST_CASE_TENSOR = [
    torch.Tensor([1.0]).to("cuda:0"),
    "cuda:0",
    "cpu",
]
TEST_CASE_LIST = [
    2 * [torch.Tensor([1.0])],
    "cpu",
    "cuda:0",
]
TEST_CASE_TUPLE = [
    2 * (torch.Tensor([1.0]),),
    "cpu",
    "cuda:0",
]
TEST_CASE_MIXED_LIST = [
    [torch.Tensor([1.0]), np.array([1])],
    "cpu",
    "cuda:0",
]
TEST_CASE_DICT = [
    {
        "x": torch.Tensor([1.0]),
        "y": 2 * [torch.Tensor([1.0])],
        "z": np.array([1]),
    },
    "cpu",
    "cuda:0",
]
TEST_CASES = [TEST_CASE_TENSOR, TEST_CASE_LIST, TEST_CASE_TUPLE, TEST_CASE_MIXED_LIST, TEST_CASE_DICT]


@skipUnless(torch.cuda.is_available(), "torch required to be built with CUDA.")
class TestCopyToDevice(unittest.TestCase):
    def _check_on_device(self, obj, device):
        if hasattr(obj, "device"):
            self.assertTrue(str(obj.device) == device)
        elif any(isinstance(obj, x) for x in [list, tuple]):
            _ = [self._check_on_device(o, device) for o in obj]
        elif isinstance(obj, dict):
            _ = [self._check_on_device(o, device) for o in obj.values()]

    @parameterized.expand(TEST_CASES)
    def test_copy(self, input, in_device, out_device):
        out = copy_to_device(input, out_device)
        self._check_on_device(input, in_device)
        self._check_on_device(out, out_device)


if __name__ == "__main__":
    unittest.main()
