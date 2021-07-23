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

from monai.transforms import KeepLargestConnectedComponent
from tests.utils import TEST_NDARRAYS

grid_1 = torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]])
grid_2 = torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 2], [1, 0, 1, 2, 2], [0, 0, 0, 0, 1]]])
grid_3 = torch.tensor(
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
    ],
)
grid_4 = torch.tensor(
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
    ],
)


TEST_CASE_1 = [
    "value_1",
    {"independent": False, "applied_labels": 1},
    grid_1,
    torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
]

TEST_CASE_2 = [
    "value_2",
    {"independent": False, "applied_labels": [2]},
    grid_1,
    torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
]

TEST_CASE_3 = [
    "independent_value_1_2",
    {"independent": True, "applied_labels": [1, 2]},
    grid_1,
    torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
]

TEST_CASE_4 = [
    "dependent_value_1_2",
    {"independent": False, "applied_labels": [1, 2]},
    grid_1,
    torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
]

TEST_CASE_5 = [
    "value_1",
    {"independent": True, "applied_labels": [1]},
    grid_2,
    torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
]

TEST_CASE_6 = [
    "independent_value_1_2",
    {"independent": True, "applied_labels": [1, 2]},
    grid_2,
    torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
]

TEST_CASE_7 = [
    "dependent_value_1_2",
    {"independent": False, "applied_labels": [1, 2]},
    grid_2,
    torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]),
]

TEST_CASE_8 = [
    "value_1_connect_1",
    {"independent": False, "applied_labels": [1], "connectivity": 1},
    grid_1,
    torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 2]]]),
]

TEST_CASE_9 = [
    "independent_value_1_2_connect_1",
    {"independent": True, "applied_labels": [1, 2], "connectivity": 1},
    grid_1,
    torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
]

TEST_CASE_10 = [
    "dependent_value_1_2_connect_1",
    {"independent": False, "applied_labels": [1, 2], "connectivity": 1},
    grid_1,
    torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
]

TEST_CASE_11 = [
    "onehot_independent_batch_2_apply_label_1_connect_1",
    {"independent": True, "applied_labels": [1], "connectivity": 1},
    grid_3,
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
        ],
    ),
]

TEST_CASE_12 = [
    "onehot_independent_batch_2_apply_label_1_connect_2",
    {"independent": True, "applied_labels": [1], "connectivity": 2},
    grid_3,
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
        ],
    ),
]

TEST_CASE_13 = [
    "onehot_independent_batch_2_apply_label_1_2_connect_2",
    {"independent": True, "applied_labels": [1, 2], "connectivity": 2},
    grid_3,
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
        ],
    ),
]

TEST_CASE_14 = [
    "onehot_dependent_batch_2_apply_label_1_2_connect_2",
    {"independent": False, "applied_labels": [1, 2], "connectivity": 2},
    grid_4,
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
        ],
    ),
]

TEST_CASE_15 = [
    "onehot_dependent_batch_2_apply_label_1_2_connect_1",
    {"independent": False, "applied_labels": [1, 2], "connectivity": 1},
    grid_4,
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
        ],
    ),
]

VALID_CASES = [
    TEST_CASE_1,
    TEST_CASE_2,
    TEST_CASE_3,
    TEST_CASE_4,
    TEST_CASE_5,
    TEST_CASE_6,
    TEST_CASE_7,
    TEST_CASE_8,
    TEST_CASE_9,
    TEST_CASE_10,
    TEST_CASE_11,
    TEST_CASE_12,
    TEST_CASE_13,
    TEST_CASE_14,
    TEST_CASE_15,
]

ITEST_CASE_1 = ["no_applied_labels_for_single_channel", {"independent": False}, grid_1, TypeError]

ITEST_CASE_2 = ["no_applied_labels_for_multi_channel", {"independent": False}, grid_3, TypeError]

INVALID_CASES = [ITEST_CASE_1, ITEST_CASE_2]


class TestKeepLargestConnectedComponent(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, args, img, expected):
        for p in TEST_NDARRAYS:
            converter = KeepLargestConnectedComponent(**args)
            im = p(img.clone())
            result = converter(im)
            self.assertEqual(type(im), type(result))
            if isinstance(result, torch.Tensor):
                self.assertEqual(result.device, im.device)
                result = result.cpu()
            np.testing.assert_allclose(result, expected)

    @parameterized.expand(INVALID_CASES)
    def test_raise_exception(self, _, args, img, expected_error):
        with self.assertRaises(expected_error):
            for p in TEST_NDARRAYS:
                converter = KeepLargestConnectedComponent(**args)
                im = p(img)
                _ = converter(im)


if __name__ == "__main__":
    unittest.main()
