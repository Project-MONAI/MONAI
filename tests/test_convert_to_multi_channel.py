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
import torch
from parameterized import parameterized

from monai.transforms import ConvertToMultiChannelBasedOnBratsClasses
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            p(np.array([[0, 1, 2], [1, 2, 4], [0, 1, 4]])),
            np.array(
                [
                    [[0, 1, 0], [1, 0, 1], [0, 1, 1]],
                    [[0, 1, 1], [1, 1, 1], [0, 1, 1]],
                    [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            p(np.array([[[[0, 1], [1, 2]], [[2, 4], [4, 4]]]])),
            np.array(
                [
                    [[[0, 1], [1, 0]], [[0, 1], [1, 1]]],
                    [[[0, 1], [1, 1]], [[1, 1], [1, 1]]],
                    [[[0, 0], [0, 0]], [[0, 1], [1, 1]]],
                ]
            ),
        ]
    )


class TestConvertToMultiChannel(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_type_shape(self, data, expected_result):
        result = ConvertToMultiChannelBasedOnBratsClasses()(data)
        self.assertEqual(type(result), type(data))
        if isinstance(result, torch.Tensor):
            self.assertEqual(result.device, data.device)
            result = result.cpu().numpy()
        np.testing.assert_equal(result, expected_result)
        self.assertEqual(f"{result.dtype}", "bool")


if __name__ == "__main__":
    unittest.main()
