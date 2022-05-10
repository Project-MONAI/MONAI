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

from monai.transforms import AsChannelFirst
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p, {"channel_dim": -1}, (4, 1, 2, 3)])
    TESTS.append([p, {"channel_dim": 3}, (4, 1, 2, 3)])
    TESTS.append([p, {"channel_dim": 2}, (3, 1, 2, 4)])


class TestAsChannelFirst(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, in_type, input_param, expected_shape):
        test_data = in_type(np.random.randint(0, 2, size=[1, 2, 3, 4]))
        result = AsChannelFirst(**input_param)(test_data)
        self.assertTupleEqual(result.shape, expected_shape)
        if isinstance(test_data, torch.Tensor):
            test_data = test_data.cpu().numpy()
        expected = np.moveaxis(test_data, input_param["channel_dim"], 0)
        assert_allclose(result, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
