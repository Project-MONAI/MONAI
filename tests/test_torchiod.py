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

from __future__ import annotations

import unittest
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import TorchIOd
from monai.utils import optional_import, set_determinism
from tests.utils import assert_allclose

_, has_torchio = optional_import("torchio")

TEST_DIMS = [3, 128, 160, 160]
TEST_TENSOR = torch.rand(TEST_DIMS)
TEST1 = [
    [
        {"keys": "img", "name": "RescaleIntensity", "out_min_max": (0, 42)},
        {"img": TEST_TENSOR},
        ((TEST_TENSOR - TEST_TENSOR.min()) / (TEST_TENSOR.max() - TEST_TENSOR.min())) * 42,
    ]
]
TEST2 = [
    [
        {"keys": ["img1", "img2"], "name": "RandomAffine", "apply_same_transform": True},
        {"img1": TEST_TENSOR, "img2": TEST_TENSOR},
    ]
]
TEST3 = [[{"keys": ["img1", "img2"], "name": "RandomAffine"}, {"img1": TEST_TENSOR, "img2": TEST_TENSOR}]]


@skipUnless(has_torchio, "Requires torchio")
class TestTorchIOd(unittest.TestCase):

    @parameterized.expand(TEST1)
    def test_value(self, input_param, input_data, expected_value):
        set_determinism(seed=0)
        result = TorchIOd(**input_param)(input_data)
        assert_allclose(result["img"], expected_value, atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand(TEST2)
    def test_common_random_transform(self, input_param, input_data):
        set_determinism(seed=0)
        result = TorchIOd(**input_param)(input_data)
        assert_allclose(result["img1"], result["img2"], atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand(TEST3)
    def test_different_random_transform(self, input_param, input_data):
        set_determinism(seed=0)
        result = TorchIOd(**input_param)(input_data)
        equal = np.allclose(result["img1"], result["img2"], atol=1e-4, rtol=1e-4)
        self.assertFalse(equal)


if __name__ == "__main__":
    unittest.main()
