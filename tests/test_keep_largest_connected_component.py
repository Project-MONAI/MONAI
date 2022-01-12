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

from monai.transforms import KeepLargestConnectedComponent
from tests.utils import TEST_NDARRAYS, assert_allclose

grid_1 = [[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]
grid_2 = [[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 2], [1, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]
grid_3 = [
    [
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ],
    [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
    ],
]
grid_4 = [
    [
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
]
grid_5 = [[[0, 0, 1, 0, 0], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1]]]

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            "value_1",
            {"independent": False, "applied_labels": 1, "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    TESTS.append(
        [
            "value_2",
            {"independent": False, "applied_labels": [2], "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "independent_value_1_2",
            {"independent": True, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "dependent_value_1_2",
            {"independent": False, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    TESTS.append(
        [
            "value_1",
            {"independent": True, "applied_labels": [1], "is_onehot": False},
            p(grid_2),
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "independent_value_1_2",
            {"independent": True, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_2),
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "dependent_value_1_2",
            {"independent": False, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_2),
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]),
        ]
    )

    TESTS.append(
        [
            "value_1_connect_1",
            {"independent": False, "applied_labels": [1], "connectivity": 1, "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    TESTS.append(
        [
            "independent_value_1_2_connect_1",
            {"independent": True, "applied_labels": [1, 2], "connectivity": 1, "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "dependent_value_1_2_connect_1",
            {"independent": False, "applied_labels": [1, 2], "connectivity": 1, "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "onehot_independent_batch_2_apply_label_1_connect_1",
            {"independent": True, "applied_labels": [1], "connectivity": 1, "is_onehot": True},
            p(grid_3),
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 1.0],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            "onehot_independent_batch_2_apply_label_1_connect_2",
            {"independent": True, "applied_labels": [1], "connectivity": 2, "is_onehot": True},
            p(grid_3),
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 1.0],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            "onehot_independent_batch_2_apply_label_1_2_connect_2",
            {"independent": True, "applied_labels": [1, 2], "connectivity": 2, "is_onehot": True},
            p(grid_3),
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            "onehot_dependent_batch_2_apply_label_1_2_connect_2",
            {"independent": False, "applied_labels": [1, 2], "connectivity": 2, "is_onehot": True},
            p(grid_4),
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            "onehot_dependent_batch_2_apply_label_1_2_connect_1",
            {"independent": False, "applied_labels": [1, 2], "connectivity": 1, "is_onehot": True},
            p(grid_4),
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            "single_channel_onehot",
            {"independent": False, "applied_labels": 0, "connectivity": 1, "is_onehot": True},
            p(grid_5),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]]),
        ]
    )

INVALID_CASES = []
for p in TEST_NDARRAYS:
    INVALID_CASES.append(
        ["no_applied_labels_for_single_channel", {"independent": False, "is_onehot": False}, p(grid_1), TypeError]
    )
    INVALID_CASES.append(
        ["no_applied_labels_for_multi_channel", {"independent": False, "is_onehot": False}, p(grid_3), TypeError]
    )
    INVALID_CASES.append(
        ["no_is_onehot_for_multi_channel", {"independent": False, "applied_labels": 0}, p(grid_3), TypeError]
    )


class TestKeepLargestConnectedComponent(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, args, input_image, expected):
        converter = KeepLargestConnectedComponent(**args)
        result = converter(input_image)
        assert_allclose(result, expected, type_test=False)

    @parameterized.expand(INVALID_CASES)
    def test_raise_exception(self, _, args, input_image, expected_error):
        with self.assertRaises(expected_error):
            converter = KeepLargestConnectedComponent(**args)
            _ = converter(input_image)


if __name__ == "__main__":
    unittest.main()
