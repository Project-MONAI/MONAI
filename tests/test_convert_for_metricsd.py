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
from monai.transforms import ConvertForMetricsd

TEST_CASE_1 = [
    {
        "keys": ["pred", "label"],
        "output_postfix": "metrics",
        "add_sigmoid": False,
        "add_softmax": False,
        "add_argmax": True,
        "to_onehot_y_pred": True,
        "to_onehot_y": True,
        "n_classes": 2,
        "round_values": False,
        "logit_thresh": 0.5
    },
    {
        "pred": torch.tensor([[[[0., 1.]], [[2., 3.]]]]),
        "label": torch.tensor([[[[0, 1]]]])
    },
    {
        "pred_metrics": torch.tensor([[[[0., 0.]], [[1., 1.]]]]),
        "label_metrics": torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    },
    (1, 2, 1, 2)
]

TEST_CASE_2 = [
    {
        "keys": ["pred", "label"],
        "output_postfix": "metrics",
        "add_sigmoid": True,
        "add_softmax": False,
        "add_argmax": False,
        "to_onehot_y_pred": False,
        "to_onehot_y": False,
        "n_classes": None,
        "round_values": True,
        "logit_thresh": 0.6
    },
    {
        "pred": torch.tensor([[[[0., 1.], [2., 3.]]]]),
        "label": torch.tensor([[[[0, 1], [1, 1]]]])
    },
    {
        "pred_metrics": torch.tensor([[[[0., 1.], [1., 1.]]]]),
        "label_metrics": torch.tensor([[[[0., 1.], [1., 1.]]]])
    },
    (1, 1, 2, 2)
]

TEST_CASE_3 = [
    {
        "keys": ["pred"],
        "output_postfix": "metrics",
        "add_sigmoid": False,
        "add_softmax": False,
        "add_argmax": True,
        "to_onehot_y_pred": True,
        "to_onehot_y": True,
        "n_classes": None,
        "round_values": False,
        "logit_thresh": 0.5
    },
    {"pred": torch.tensor([[[[0., 1.]], [[2., 3.]]]])},
    {"pred_metrics": torch.tensor([[[[0., 0.]], [[1., 1.]]]])},
    (1, 2, 1, 2)
]


class TestConvertForMetricsd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_param, test_input, output, expected_shape):
        result = ConvertForMetricsd(**input_param)(test_input)
        torch.testing.assert_allclose(result["pred_metrics"], output["pred_metrics"])
        self.assertTupleEqual(result["pred_metrics"].shape, expected_shape)
        if "label_metrics" in result:
            torch.testing.assert_allclose(result["label_metrics"], output["label_metrics"])
            self.assertTupleEqual(result["label_metrics"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
