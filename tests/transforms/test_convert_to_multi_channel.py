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

from __future__ import annotations

import unittest

import torch
from parameterized import parameterized

from monai.transforms import ConvertToMultiChannelBasedOnBratsClasses
from tests.test_utils import TEST_NDARRAYS, assert_allclose

TESTS = []
TESTS_ET_LABEL_3 = []

# Tests for default et_label = 4
for p in TEST_NDARRAYS:
    TESTS.extend(
        [
            [
                p([[0, 1, 2], [1, 2, 4], [0, 1, 4]]),
                p(
                    [
                        [[0, 1, 0], [1, 0, 1], [0, 1, 1]],
                        [[0, 1, 1], [1, 1, 1], [0, 1, 1]],
                        [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
                    ]
                ),
            ],
            [
                p([[[[0, 1], [1, 2]], [[2, 4], [4, 4]]]]),
                p(
                    [
                        [[[0, 1], [1, 0]], [[0, 1], [1, 1]]],
                        [[[0, 1], [1, 1]], [[1, 1], [1, 1]]],
                        [[[0, 0], [0, 0]], [[0, 1], [1, 1]]],
                    ]
                ),
            ],
        ]
    )

# Tests for et_label = 3
for p in TEST_NDARRAYS:
    TESTS_ET_LABEL_3.extend(
        [
            [
                p([[0, 1, 2], [1, 2, 3], [0, 1, 3]]),
                p(
                    [
                        [[0, 1, 0], [1, 0, 1], [0, 1, 1]],
                        [[0, 1, 1], [1, 1, 1], [0, 1, 1]],
                        [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
                    ]
                ),
            ]
        ]
    )


class TestConvertToMultiChannel(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_type_shape(self, data, expected_result):
        result = ConvertToMultiChannelBasedOnBratsClasses()(data)
        assert_allclose(result, expected_result)
        self.assertTrue(result.dtype in (bool, torch.bool))

    @parameterized.expand(TESTS_ET_LABEL_3)
    def test_type_shape_et_label_3(self, data, expected_result):
        result = ConvertToMultiChannelBasedOnBratsClasses(et_label=3)(data)
        assert_allclose(result, expected_result)
        self.assertTrue(result.dtype in (bool, torch.bool))

    def test_invalid_et_label(self):
        with self.assertRaises(ValueError):
            ConvertToMultiChannelBasedOnBratsClasses(et_label=1)
        with self.assertRaises(ValueError):
            ConvertToMultiChannelBasedOnBratsClasses(et_label=2)


if __name__ == "__main__":
    unittest.main()
