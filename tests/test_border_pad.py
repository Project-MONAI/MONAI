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
from monai.transforms import BorderPad
from monai.utils import NumpyPadMode
from tests.utils import TEST_NDARRAYS, assert_allclose

TEST_CASE_1 = [{"spatial_border": 2, "mode": "constant"}, np.zeros((3, 8, 8, 4)), np.zeros((3, 12, 12, 8))]

TEST_CASE_2 = [{"spatial_border": [1, 2, 3], "mode": "constant"}, np.zeros((3, 8, 8, 4)), np.zeros((3, 10, 12, 10))]

TEST_CASE_3 = [
    {"spatial_border": [1, 2, 3, 4, 5, 6], "mode": "constant"},
    np.zeros((3, 8, 8, 4)),
    np.zeros((3, 11, 15, 15)),
]

TEST_CASE_4 = [
    {"spatial_border": [1, 2, 3, 4, 5, 6], "mode": NumpyPadMode.CONSTANT},
    np.zeros((3, 8, 8, 4)),
    np.zeros((3, 11, 15, 15)),
]


class TestBorderPad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_pad_shape(self, input_param, input_data, expected_val):
        base_comparison = None
        padder = BorderPad(**input_param)
        # check result is the same regardless of input type
        for p in TEST_NDARRAYS:
            r1 = padder(p(input_data))
            r2 = padder(input_data, mode=input_param["mode"])
            # check shape
            self.assertAlmostEqual(r1.shape, expected_val.shape)
            self.assertAlmostEqual(r2.shape, expected_val.shape)
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
            padder = BorderPad(spatial_border=2, mode="constant", **kwargs)
            result = padder(im)
            if isinstance(result, torch.Tensor):
                result = result.cpu()
            # central section should remain unchanged
            assert_allclose(result[:, 2:-2, 2:-2], im, type_test=False)
            expected_vals = [0, 2] if isinstance(im, torch.Tensor) else [0, 1, 2]
            assert_allclose(np.unique(result), expected_vals, type_test=False)
            # check inverse
            if isinstance(result, MetaTensor):
                inv = padder.inverse(result)
                assert_allclose(im, inv, type_test=False)
                self.assertEqual(inv.applied_operations, [])


if __name__ == "__main__":
    unittest.main()
