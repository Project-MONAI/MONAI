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

from monai.losses import MCCLoss
from tests.test_utils import test_script_save

TEST_CASES = [
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2), sigmoid
        {"include_background": True, "sigmoid": True},
        {"input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])},
        0.733197,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2), sigmoid
        {"include_background": True, "sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        1.0,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2), perfect prediction
        {"include_background": True},
        {"input": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])},
        0.0,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2), worst case (inverted)
        {"include_background": True},
        {"input": torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]), "target": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])},
        2.0,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3), multi-class, exclude background, one-hot
        {"include_background": False, "to_onehot_y": True},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
        },
        0.0,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3), multi-class, sigmoid, one-hot
        {"include_background": True, "to_onehot_y": True, "sigmoid": True},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        0.836617,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3), multi-class, sigmoid, one-hot, batch=True
        {"include_background": True, "to_onehot_y": True, "sigmoid": True, "batch": True},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        0.845961,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3), multi-class, sigmoid, one-hot, reduction=sum
        {"include_background": True, "to_onehot_y": True, "sigmoid": True, "reduction": "sum"},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        3.346468,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3), multi-class, softmax, one-hot
        {"include_background": True, "to_onehot_y": True, "softmax": True},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        0.730736,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3), multi-class, softmax, one-hot, reduction=none
        {"include_background": True, "to_onehot_y": True, "softmax": True, "reduction": "none"},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
        },
        [[0.461472, 0.461472], [1.0, 1.0]],
    ],
    [  # shape: (1, 1, 3, 3), (1, 1, 3, 3), all-ones perfect prediction
        {"include_background": True},
        {"input": torch.ones(1, 1, 3, 3), "target": torch.ones(1, 1, 3, 3)},
        0.0,
    ],
    [  # shape: (1, 1, 3, 3), (1, 1, 3, 3), all-zeros perfect prediction
        {"include_background": True},
        {"input": torch.zeros(1, 1, 3, 3), "target": torch.zeros(1, 1, 3, 3)},
        0.0,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2), other_act=torch.tanh
        {"include_background": True, "other_act": torch.tanh},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
        },
        1.0,
    ],
]


class TestMCCLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = MCCLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-4)

    def test_ill_shape(self):
        loss = MCCLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((2, 2, 3)), torch.ones((4, 5, 6)))
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            MCCLoss(reduction="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            MCCLoss(reduction=None)(chn_input, chn_target)

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            MCCLoss(sigmoid=True, softmax=True)
        with self.assertRaisesRegex(TypeError, ""):
            MCCLoss(other_act="tanh")

    @parameterized.expand([(False, False, False), (False, True, False), (False, False, True)])
    def test_input_warnings(self, include_background, softmax, to_onehot_y):
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertWarns(Warning):
            loss = MCCLoss(include_background=include_background, softmax=softmax, to_onehot_y=to_onehot_y)
            loss.forward(chn_input, chn_target)

    def test_script(self):
        loss = MCCLoss()
        test_input = torch.ones(2, 1, 8, 8)
        test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
