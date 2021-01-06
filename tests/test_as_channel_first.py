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

import numpy as np
from parameterized import parameterized

from monai.transforms import AsChannelFirst

TEST_CASE_1 = [{"channel_dim": -1}, (4, 1, 2, 3)]

TEST_CASE_2 = [{"channel_dim": 3}, (4, 1, 2, 3)]

TEST_CASE_3 = [{"channel_dim": 2}, (3, 1, 2, 4)]


class TestAsChannelFirst(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_param, expected_shape):
        test_data = np.random.randint(0, 2, size=[1, 2, 3, 4])
        result = AsChannelFirst(**input_param)(test_data)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
