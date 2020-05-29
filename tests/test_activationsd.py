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
from monai.transforms import Activationsd

TEST_CASE_1 = [
    {"keys": ["pred", "label"], "output_postfix": "act", "sigmoid": False, "softmax": [True, False], "other": None},
    {"pred": torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]), "label": torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]])},
    {
        "pred_act": torch.tensor([[[[0.1192, 0.1192]], [[0.8808, 0.8808]]]]),
        "label_act": torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
    },
    (1, 2, 1, 2),
]

TEST_CASE_2 = [
    {
        "keys": ["pred", "label"],
        "output_postfix": "act",
        "sigmoid": False,
        "softmax": False,
        "other": [lambda x: torch.tanh(x), None],
    },
    {"pred": torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]), "label": torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])},
    {
        "pred_act": torch.tensor([[[[0.0000, 0.7616], [0.9640, 0.9951]]]]),
        "label_act": torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
    },
    (1, 1, 2, 2),
]


class TestActivationsd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, test_input, output, expected_shape):
        result = Activationsd(**input_param)(test_input)
        torch.testing.assert_allclose(result["pred_act"], output["pred_act"])
        self.assertTupleEqual(result["pred_act"].shape, expected_shape)
        if "label_act" in result:
            torch.testing.assert_allclose(result["label_act"], output["label_act"])
            self.assertTupleEqual(result["label_act"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
