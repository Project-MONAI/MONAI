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
import torch
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import DivisiblePad
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []

# pad first dim to be divisible by 7, the second unchanged.
TESTS.append([{"k": (7, -1), "mode": "constant"}, (3, 8, 7), (3, 14, 7)])
# pad all dimensions to be divisible by 5
TESTS.append([{"k": 5, "mode": "constant", "method": "end"}, (3, 10, 5, 17), (3, 10, 5, 20)])


class TestDivisiblePad(unittest.TestCase):
    @staticmethod
    def get_arr(shape):
        return np.random.randint(100, size=shape).astype(float)

    @parameterized.expand(TESTS)
    def test_pad_shape(self, input_param, input_shape, expected_shape):
        base_comparison = None
        input_data = self.get_arr(input_shape)
        padder = DivisiblePad(**input_param)
        # check result is the same regardless of input type
        for p in TEST_NDARRAYS:
            r1 = padder(p(input_data))
            r2 = padder(p(input_data), mode=input_param["mode"])
            # check
            np.testing.assert_allclose(r1.shape, expected_shape)
            np.testing.assert_allclose(r2.shape, expected_shape)
            # check results are same regardless of input type
            if base_comparison is None:
                base_comparison = r1
            torch.testing.assert_allclose(r1, base_comparison, atol=0, rtol=1e-5)
            torch.testing.assert_allclose(r2, base_comparison, atol=0, rtol=1e-5)
            # test inverse
            for r in (r1, r2):
                if isinstance(r, MetaTensor):
                    r = padder.inverse(r)
                    self.assertIsInstance(r, MetaTensor)
                    assert_allclose(r, input_data, type_test=False)
                    self.assertEqual(r.applied_operations, [])

    def test_pad_kwargs(self):
        for p in TEST_NDARRAYS:
            im = p(np.zeros((3, 8, 4)))
            kwargs = {"value": 2} if isinstance(im, torch.Tensor) else {"constant_values": ((0, 0), (1, 1), (2, 2))}
            padder = DivisiblePad(k=5, method="end", mode="constant", **kwargs)
            result = padder(im)
            if isinstance(result, torch.Tensor):
                result = result.cpu()
            expected_vals = [0, 2] if isinstance(im, torch.Tensor) else [0, 1, 2]
            assert_allclose(np.unique(result), expected_vals, type_test=False)
            # check inverse
            if isinstance(result, MetaTensor):
                inv = padder.inverse(result)
                assert_allclose(im, inv, type_test=False)
                self.assertEqual(inv.applied_operations, [])


if __name__ == "__main__":
    unittest.main()
