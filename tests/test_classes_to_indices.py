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

from monai.transforms import ClassesToIndices
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            # test Argmax data
            {"num_classes": 3, "image_threshold": 0.0},
            p(np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
            None,
            [np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7])],
        ]
    )

    TESTS.append(
        [
            {"num_classes": 3, "image_threshold": 60},
            p(np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
            p(np.array([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]])),
            [np.array([0, 8]), np.array([1, 5, 6]), np.array([3])],
        ]
    )

    TESTS.append(
        [
            # test One-Hot data
            {"image_threshold": 0.0},
            p(
                np.array(
                    [
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                    ]
                )
            ),
            None,
            [np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7])],
        ]
    )

    TESTS.append(
        [
            {"num_classes": None, "image_threshold": 60},
            p(
                np.array(
                    [
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                    ]
                )
            ),
            p(np.array([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]])),
            [np.array([0, 8]), np.array([1, 5, 6]), np.array([3])],
        ]
    )

    TESTS.append(
        [
            # test output_shape
            {"num_classes": 3, "image_threshold": 0.0, "output_shape": [3, 3]},
            p(np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
            None,
            [
                np.array([[0, 0], [1, 1], [2, 2]]),
                np.array([[0, 1], [1, 2], [2, 0]]),
                np.array([[0, 2], [1, 0], [2, 1]]),
            ],
        ]
    )


class TestClassesToIndices(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_args, label, image, expected_indices):
        indices = ClassesToIndices(**input_args)(label, image)
        for i, e in zip(indices, expected_indices):
            i = i.cpu() if isinstance(i, torch.Tensor) else i
            np.testing.assert_allclose(i, e)


if __name__ == "__main__":
    unittest.main()
