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
from parameterized import parameterized

from monai.transforms import SplitChannel
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([{"channel_dim": 1}, p(np.random.randint(2, size=(4, 3, 3, 4))), (4, 1, 3, 4)])
    TESTS.append([{"channel_dim": 0}, p(np.random.randint(2, size=(3, 3, 4))), (1, 3, 4)])
    TESTS.append([{"channel_dim": 2}, p(np.random.randint(2, size=(3, 2, 4))), (3, 2, 1)])
    TESTS.append([{"channel_dim": -1}, p(np.random.randint(2, size=(3, 2, 4))), (3, 2, 1)])


class TestSplitChannel(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, input_param, test_data, expected_shape):
        result = SplitChannel(**input_param)(test_data)
        for data in result:
            self.assertEqual(type(data), type(test_data))
            self.assertTupleEqual(data.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
