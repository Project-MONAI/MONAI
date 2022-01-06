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
from parameterized import parameterized

from monai.utils import sample_slices
from tests.utils import TEST_NDARRAYS, assert_allclose

TEST_CASE_1 = [torch.tensor([[[0, 2], [1, 0]]]), 1, (1,), (1, 1, 2), torch.tensor([[[1, 0]]])]

TEST_CASE_2 = [torch.tensor([[[0, 2], [1, 0], [4, 5]]]), 1, (0, 2), (1, 2, 2), torch.tensor([[[0, 2], [4, 5]]])]


class TestSampleSlices(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_data, dim, vals, expected_shape, expected_result):
        for p in TEST_NDARRAYS:
            result = sample_slices(p(input_data), dim, *vals)

            self.assertEqual(result.shape, expected_shape)
            assert_allclose(p(expected_result), result)


if __name__ == "__main__":
    unittest.main()
