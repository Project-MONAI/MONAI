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

from monai.losses import HausdorffDTLoss

TEST_CASES = [
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "sigmoid": True},
        {"input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])},
        0.509329,
    ],
    [  # shape: (1, 2, 2, 2), (1, 2, 2, 2)
        {"include_background": True, "sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
        },
        0.326994,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        0.758470,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": False, "sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]),
        },
        0.144659,
    ],
    [  # shape: (2, 2, 3, 1), (2, 1, 3, 1)
        {"include_background": True, "to_onehot_y": True, "sigmoid": True, "reduction": "none"},
        {
            "input": torch.tensor(
                [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]]
            ),
            "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]]),
        },
        [[[[0.407765]], [[0.407765]]], [[[0.5000]], [[0.5000]]]],
    ],
    [  # shape: (2, 2, 3, 1), (2, 1, 3, 1)
        {"include_background": True, "to_onehot_y": True, "softmax": True},
        {
            "input": torch.tensor(
                [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]]
            ),
            "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]]),
        },
        0.357016,
    ],
    [  # shape: (2, 2, 3, 1), (2, 1, 3, 1)
        {"include_background": True, "to_onehot_y": True, "softmax": True, "reduction": "sum"},
        {
            "input": torch.tensor(
                [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]]
            ),
            "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]]),
        },
        1.428062,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "sigmoid": True},
        {"input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])},
        0.509329,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "other_act": torch.tanh},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        3.450064,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "other_act": lambda x: torch.log_softmax(x, dim=1)},
        {
            "input": torch.tensor(
                [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]]
            ),
            "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]]),
        },
        4.366613,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "other_act": torch.tanh, "batch": True},
        {
            "input": torch.tensor([[[[1.0, -0.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        2.661359,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "other_act": torch.tanh, "batch": True},
        {
            "input": torch.tensor([[[[1.0, -0.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        2.661359,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "other_act": torch.tanh, "batch": False},
        {
            "input": torch.tensor([[[[1.0, -0.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        2.661359,
    ],
]


class TestHausdorffDTLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = HausdorffDTLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = HausdorffDTLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 1, 2, 3)), torch.ones((1, 4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            HausdorffDTLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            HausdorffDTLoss(reduction="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            HausdorffDTLoss(reduction=None)(chn_input, chn_target)

    def test_input_warnings(self):
        chn_input = torch.ones((1, 1, 1, 3))
        chn_target = torch.ones((1, 1, 1, 3))
        with self.assertWarns(Warning):
            loss = HausdorffDTLoss(include_background=False)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = HausdorffDTLoss(softmax=True)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = HausdorffDTLoss(to_onehot_y=True)
            loss.forward(chn_input, chn_target)


if __name__ == "__main__":
    unittest.main()