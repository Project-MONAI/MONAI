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

from mri_transforms import (
    apply_mask,
    complex_abs,
    convert_to_tensor_complex,
    create_mask_for_mask_type,
)
from monai.utils.type_conversion import convert_data_type
from tests.utils import TEST_NDARRAYS, assert_allclose

# test case for convert_to_tensor_complex
im = [[1.0, 1.0], [1.0, 1.0]]
im_complex = [[1.0 + 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 + 1.0j]]
TESTS = [(np.array(im), (2, 2)), (np.array(im_complex), (2, 2, 2))]

# test case for complex_abs
im = [[3.0, 4.0], [3.0, 4.0]]
res = [5.0, 5.0]
TESTSC = []
for p in TEST_NDARRAYS:
    TESTSC.append((p(im), p(res)))

# test case for apply_mask
ksp, *_ = convert_data_type(np.ones([50, 50, 2]), torch.Tensor)
TESTSM = [(ksp,)]


class TestMRIUtils(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_to_tensor_complex(self, test_data, expected_shape):
        result = convert_to_tensor_complex(test_data)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand(TESTSC)
    def test_complex_abs(self, test_data, res_data):
        result = complex_abs(test_data)
        assert_allclose(result, res_data, type_test=False)

    @parameterized.expand(TESTSM)
    def test_mask(self, test_data):
        # random mask function
        mask_func = create_mask_for_mask_type(mask_type_str="random", center_fractions=[0.08], accelerations=[4.0])
        result, mask = apply_mask(test_data, mask_func)
        result = result[..., mask.squeeze() == 0, :].sum()
        self.assertEqual(result.item(), 0)

        # equispaced mask function
        mask_func = create_mask_for_mask_type(mask_type_str="equispaced", center_fractions=[0.08], accelerations=[4.0])
        result, mask = apply_mask(test_data, mask_func)
        result = result[..., mask.squeeze() == 0, :].sum()
        self.assertEqual(result.item(), 0)


if __name__ == "__main__":
    unittest.main()
