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

from parameterized import parameterized

from monai.transforms import ToTensor
from tests.utils import TEST_NDARRAYS, assert_allclose

im = [[1, 2], [3, 4]]

TESTS = []
TESTS.append((im, (2, 2)))
for p in TEST_NDARRAYS:
    TESTS.append((p(im), (2, 2)))

TESTS_SINGLE = []
TESTS_SINGLE.append([5])
for p in TEST_NDARRAYS:
    TESTS_SINGLE.append([p(5)])


class TestToTensor(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_array_input(self, test_data, expected_shape):
        result = ToTensor()(test_data)
        assert_allclose(result, test_data)
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand(TESTS_SINGLE)
    def test_single_input(self, test_data):
        result = ToTensor()(test_data)
        assert_allclose(result, test_data)
        self.assertEqual(result.ndim, 0)


if __name__ == "__main__":
    unittest.main()
