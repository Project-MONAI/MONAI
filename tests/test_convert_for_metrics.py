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
from monai.transforms import ConvertForMetrics

TEST_CASE_1 = [
    {
        "add_sigmoid": True,
        "add_softmax": False,
        "add_argmax": False,
        "to_onehot_y_pred": False,
        "to_onehot_y": False,
        "n_classes": None,
        "round_values": False,
        "logit_thresh": 0.5,
    },
    torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
    torch.tensor([[[[0, 1], [1, 1]]]]),
    torch.tensor([[[[0.5000, 0.7311], [0.8808, 0.9526]]]]),
    torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]]),
    (1, 1, 2, 2),
]

TEST_CASE_2 = [
    {
        "add_sigmoid": False,
        "add_softmax": True,
        "add_argmax": False,
        "to_onehot_y_pred": False,
        "to_onehot_y": False,
        "n_classes": None,
        "round_values": False,
        "logit_thresh": 0.5,
    },
    torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
    torch.tensor([[[[0, 1]], [[1, 0]]]]),
    torch.tensor([[[[0.1192, 0.1192]], [[0.8808, 0.8808]]]]),
    torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]]),
    (1, 2, 1, 2),
]

TEST_CASE_3 = [
    {
        "add_sigmoid": False,
        "add_softmax": False,
        "add_argmax": True,
        "to_onehot_y_pred": False,
        "to_onehot_y": False,
        "n_classes": None,
        "round_values": False,
        "logit_thresh": 0.5,
    },
    torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
    torch.tensor([[[[0, 1]]]]),
    torch.tensor([[[[1.0, 1.0]]]]),
    torch.tensor([[[[0.0, 1.0]]]]),
    (1, 1, 1, 2),
]

TEST_CASE_4 = [
    {
        "add_sigmoid": False,
        "add_softmax": False,
        "add_argmax": True,
        "to_onehot_y_pred": True,
        "to_onehot_y": True,
        "n_classes": None,
        "round_values": False,
        "logit_thresh": 0.5,
    },
    torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
    torch.tensor([[[[0, 1]]]]),
    torch.tensor([[[[0.0, 0.0]], [[1.0, 1.0]]]]),
    torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
    (1, 2, 1, 2),
]

TEST_CASE_5 = [
    {
        "add_sigmoid": False,
        "add_softmax": False,
        "add_argmax": True,
        "to_onehot_y_pred": True,
        "to_onehot_y": True,
        "n_classes": 2,
        "round_values": False,
        "logit_thresh": 0.5,
    },
    torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
    torch.tensor([[[[0, 1]]]]),
    torch.tensor([[[[0.0, 0.0]], [[1.0, 1.0]]]]),
    torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
    (1, 2, 1, 2),
]

TEST_CASE_6 = [
    {
        "add_sigmoid": True,
        "add_softmax": False,
        "add_argmax": False,
        "to_onehot_y_pred": False,
        "to_onehot_y": False,
        "n_classes": None,
        "round_values": True,
        "logit_thresh": 0.6,
    },
    torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
    torch.tensor([[[[0, 1], [1, 1]]]]),
    torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]]),
    torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]]),
    (1, 1, 2, 2),
]

TEST_CASE_7 = [
    {
        "add_sigmoid": False,
        "add_softmax": False,
        "add_argmax": True,
        "to_onehot_y_pred": True,
        "to_onehot_y": True,
        "n_classes": None,
        "round_values": False,
        "logit_thresh": 0.5,
    },
    torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
    None,
    torch.tensor([[[[0.0, 0.0]], [[1.0, 1.0]]]]),
    None,
    (1, 2, 1, 2),
]


class TestConvertForMetrics(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_shape(self, input_param, test_y_pred, test_y, y_pred_out, y_out, expected_shape):
        result = ConvertForMetrics(**input_param)(test_y_pred, test_y)
        torch.testing.assert_allclose(result[0], y_pred_out)
        (result[1] is None and y_out is None) or torch.testing.assert_allclose(result[1], y_out)
        for data in result:
            if data is not None:
                self.assertTupleEqual(data.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
