# Copyright 2020 MONAI Consortium
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

grid_1 = torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]])
grid_2 = torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 2], [1, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]])
grid_3 = torch.tensor(
    [
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
    ]
)


TEST_CASE_1 = [
    "value_1",
    {"independent": False, "applied_labels": 1},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_2 = [
    "value_2",
    {"independent": False, "applied_labels": [2]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_3 = [
    "independent_value_1_2",
    {"independent": True, "applied_labels": [1, 2]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_4 = [
    "dependent_value_1_2",
    {"independent": False, "applied_labels": [1, 2]},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_5 = [
    "value_1",
    {"independent": True, "applied_labels": [1]},
    grid_2,
    torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]]),
]

TEST_CASE_6 = [
    "independent_value_1_2",
    {"independent": True, "applied_labels": [1, 2]},
    grid_2,
    torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]]),
]

TEST_CASE_7 = [
    "dependent_value_1_2",
    {"independent": False, "applied_labels": [1, 2]},
    grid_2,
    torch.tensor([[[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]]),
]

TEST_CASE_8 = [
    "value_1_connect_1",
    {"independent": False, "applied_labels": [1], "connectivity": 1},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_9 = [
    "independent_value_1_2_connect_1",
    {"independent": True, "applied_labels": [1, 2], "connectivity": 1},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_10 = [
    "dependent_value_1_2_connect_1",
    {"independent": False, "applied_labels": [1, 2], "connectivity": 1},
    grid_1,
    torch.tensor([[[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]]),
]

TEST_CASE_11 = [
    "value_0_background_3",
    {"independent": False, "applied_labels": [0], "background": 3},
    grid_1,
    torch.tensor([[[[3, 3, 1, 3, 3], [3, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]]),
]

TEST_CASE_12 = [
    "all_0_batch_2",
    {"independent": False, "applied_labels": [1], "background": 3},
    torch.tensor(
        [
            [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
            [[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]],
        ]
    ),
    torch.tensor(
        [
            [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
            [[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 3, 3, 3], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0]]],
        ]
    ),
]

TEST_CASE_13 = [
    "onehot_independent_batch_2_apply_label_1_connect_1",
    {"independent": True, "applied_labels": [1], "connectivity": 1},
    grid_3,
    torch.tensor(
        [
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
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
        ]
    ),
]

TEST_CASE_14 = [
    "onehot_independent_batch_2_apply_label_1_connect_2",
    {"independent": True, "applied_labels": [1], "connectivity": 2},
    grid_3,
    torch.tensor(
        [
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
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
        ]
    ),
]

TEST_CASE_15 = [
    "onehot_independent_batch_2_apply_label_1_2_connect_2",
    {"independent": True, "applied_labels": [1, 2], "connectivity": 2},
    grid_3,
    torch.tensor(
        [
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
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
        ]
    ),
]

TEST_CASE_16 = [
    "onehot_dependent_batch_2_apply_label_1_2_connect_2",
    {"independent": False, "applied_labels": [1, 2], "connectivity": 2},
    grid_3,
    torch.tensor(
        [
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
        ]
    ),
]

TEST_CASE_17 = [
    "onehot_dependent_batch_2_apply_label_1_2_connect_1",
    {"independent": False, "applied_labels": [1, 2], "connectivity": 1},
    grid_3,
    torch.tensor(
        [
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
                    [1.0, 0.0, 0.0, 0.0, 0.0],
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
        ]
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
    TEST_CASE_16,
    TEST_CASE_17,
]

ITEST_CASE_1 = [
    "no_applied_labels_for_single_channel",
    {"independent": False},
    grid_1,
    ValueError,
]

ITEST_CASE_2 = [
    "background_in_applied_labels",
    {"independent": False, "background": 1, "applied_labels": [1, 2]},
    grid_1,
    ValueError,
]

ITEST_CASE_3 = [
    "no_applied_labels_for_multi_channel",
    {"independent": False, "background": 0},
    grid_3,
    ValueError,
]

INVALID_CASES = [
    ITEST_CASE_1,
    ITEST_CASE_2,
    ITEST_CASE_3,
]


class TestKeepLargestConnectedComponent(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, args, tensor, expected):
        converter = KeepLargestConnectedComponent(**args)
        if torch.cuda.is_available():
            result = converter(tensor.clone().cuda())
            assert torch.allclose(result, expected.cuda())
        else:
            result = converter(tensor.clone())
            assert torch.allclose(result, expected)

    @parameterized.expand(INVALID_CASES)
    def test_raise_exception(self, _, args, tensor, expected_error):
        with self.assertRaises(expected_error):
            converter = KeepLargestConnectedComponent(**args)
            if torch.cuda.is_available():
                result = converter(tensor.clone().cuda())
            else:
                result = converter(tensor.clone())


if __name__ == "__main__":
    unittest.main()
