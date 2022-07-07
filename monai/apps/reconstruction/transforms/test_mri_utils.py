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

import torch
from mri_utils import complex_abs, convert_to_tensor_complex, rss
from parameterized import parameterized

from monai.utils.type_conversion import convert_data_type
from tests.utils import TEST_NDARRAYS, assert_allclose

# test case for convert_to_tensor_complex
im_complex = [[1.0 + 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 + 1.0j]]
expected_shape = convert_data_type((2, 2, 2), torch.Tensor)[0]
TESTS = [(im_complex, expected_shape)]
for p in TEST_NDARRAYS:
    TESTS.append((p(im_complex), expected_shape))

# test case for complex_abs
im = [[3.0, 4.0], [3.0, 4.0]]
res = [5.0, 5.0]
TESTSC = []
for p in TEST_NDARRAYS:
    TESTSC.append((p(im), p(res)))


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

    @parameterized.expand(TESTSC)
    def test_rss(self, test_data, res_data):
        result = rss(test_data, spatial_dim=1)
        assert_allclose(result, res_data, type_test=False)


if __name__ == "__main__":
    unittest.main()
