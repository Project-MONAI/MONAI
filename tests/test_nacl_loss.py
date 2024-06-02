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

import numpy as np
import torch
from parameterized import parameterized

from monai.losses import NACLLoss

TEST_CASES = [
    [  # shape: (2, 2, 3), (2, 2, 3)
        {"classes": 2},
        {
            "inputs": torch.tensor(
                [
                    [[0.8959, 0.7435, 0.4429], [0.6038, 0.5506, 0.3869], [0.8485, 0.4703, 0.8790]],
                    [[0.5137, 0.8345, 0.2821], [0.3644, 0.8000, 0.5156], [0.4732, 0.2018, 0.4564]],
                ]
            ),
            "targets": torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
        },
        3.3611,  # the result equals to -1 + np.log(1 + np.exp(1))
    ],
    [  # shape: (2, 2, 3), (2, 2, 3)
        {"classes": 2, "kernel_ops": "gaussian"},
        {
            "inputs": torch.tensor(
                [
                    [[0.8959, 0.7435, 0.4429], [0.6038, 0.5506, 0.3869], [0.8485, 0.4703, 0.8790]],
                    [[0.5137, 0.8345, 0.2821], [0.3644, 0.8000, 0.5156], [0.4732, 0.2018, 0.4564]],
                ]
            ),
            "targets": torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
        },
        3.3963,  # the result equals to -1 + np.log(1 + np.exp(1))
    ],
    [  # shape: (2, 2, 3), (2, 2, 3)
        {"classes": 2, "distance_type": "l2"},
        {
            "inputs": torch.tensor(
                [
                    [[0.8959, 0.7435, 0.4429], [0.6038, 0.5506, 0.3869], [0.8485, 0.4703, 0.8790]],
                    [[0.5137, 0.8345, 0.2821], [0.3644, 0.8000, 0.5156], [0.4732, 0.2018, 0.4564]],
                ]
            ),
            "targets": torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
        },
        3.3459,  # the result equals to -1 + np.log(1 + np.exp(1))
    ],
    [  # shape: (2, 2, 3), (2, 2, 3)
        {"classes": 2, "alpha": 0.2},
        {
            "inputs": torch.tensor(
                [
                    [[0.8959, 0.7435, 0.4429], [0.6038, 0.5506, 0.3869], [0.8485, 0.4703, 0.8790]],
                    [[0.5137, 0.8345, 0.2821], [0.3644, 0.8000, 0.5156], [0.4732, 0.2018, 0.4564]],
                ]
            ),
            "targets": torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
        },
        3.3836,  # the result equals to -1 + np.log(1 + np.exp(1))
    ],
]


class TestNACLLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        loss = NACLLoss(**input_param)
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
