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

from parameterized import parameterized

from monai.transforms import GaussianSmooth
from tests.test_utils import TEST_NDARRAYS, assert_allclose

TESTS = []

for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"sigma": 1.5},
            p([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
            p(
                [
                    [
                        [0.59167546, 0.69312394, 0.59167546],
                        [0.7956997, 0.93213004, 0.7956997],
                        [0.7668002, 0.8982755, 0.7668002],
                    ],
                    [
                        [1.6105323, 1.8866735, 1.6105323],
                        [1.9892492, 2.3303251, 1.9892492],
                        [1.7856569, 2.091825, 1.7856569],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            {"sigma": 0.5},
            p([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
            p(
                [
                    [
                        [0.8424794, 0.99864554, 0.8424794],
                        [1.678146, 1.9892154, 1.678146],
                        [1.9889624, 2.3576462, 1.9889624],
                    ],
                    [
                        [2.966061, 3.5158648, 2.966061],
                        [4.1953645, 4.973038, 4.1953645],
                        [4.112544, 4.8748655, 4.1125436],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            {"sigma": [1.5, 0.5]},
            p([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
            p(
                [
                    [
                        [0.8542037, 1.0125432, 0.8542037],
                        [1.1487541, 1.3616928, 1.1487541],
                        [1.1070318, 1.3122368, 1.1070318],
                    ],
                    [
                        [2.3251305, 2.756128, 2.3251305],
                        [2.8718853, 3.4042323, 2.8718853],
                        [2.5779586, 3.0558217, 2.5779586],
                    ],
                ]
            ),
        ]
    )


class TestGaussianSmooth(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, arguments, image, expected_data):
        result = GaussianSmooth(**arguments)(image)
        assert_allclose(result, expected_data, atol=1e-4, rtol=1e-4, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
