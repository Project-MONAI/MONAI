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

from monai.transforms import AsChannelFirst
from monai.transforms.utils_pytorch_numpy_unification import moveaxis
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p, {"channel_dim": -1}, (4, 1, 2, 3)])
    TESTS.append([p, {"channel_dim": 3}, (4, 1, 2, 3)])
    TESTS.append([p, {"channel_dim": 2}, (3, 1, 2, 4)])
    TESTS.append([p, {"channel_dim": (1, 2)}, (2, 3, 1, 4)])


class TestAsChannelFirst(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, in_type, input_param, expected_shape):
        test_data = in_type(np.random.randint(0, 2, size=[1, 2, 3, 4]))
        result = AsChannelFirst(**input_param)(test_data)
        self.assertTupleEqual(result.shape, expected_shape)
        if isinstance(input_param["channel_dim"], int):
            expected = moveaxis(test_data, input_param["channel_dim"], 0)
        else:  # sequence
            expected = moveaxis(test_data, input_param["channel_dim"], tuple(range(len(input_param["channel_dim"]))))
        assert_allclose(result, expected, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
