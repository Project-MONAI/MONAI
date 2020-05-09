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

import numpy as np
import torch
from parameterized import parameterized

from monai.losses import TverskyLoss

TEST_CASES = [
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "pred": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "ground": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-6,
        },
        0.307576,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "pred": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "ground": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
            "smooth": 1e-4,
        },
        0.416657,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": False, "to_onehot_y": True},
        {
            "pred": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "ground": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
            "smooth": 0.0,
        },
        0.0,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_sigmoid": True},
        {
            "pred": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "ground": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        0.435050,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_sigmoid": True, "reduction": "sum"},
        {
            "pred": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "ground": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        1.74013,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True},
        {
            "pred": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "ground": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        0.383713,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True, "reduction": "none"},
        {
            "pred": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "ground": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        [[0.210961, 0.295339], [0.599952, 0.428547]],
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True, "alpha": 0.3, "beta": 0.7},
        {
            "pred": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "ground": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-6,
        },
        0.3589,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True, "alpha": 0.7, "beta": 0.3},
        {
            "pred": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "ground": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-6,
        },
        0.247366,
    ],
]


class TestTverskyLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = TverskyLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
