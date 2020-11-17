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

from monai.networks.layers.factories import Act
from monai.transforms import Activations

TEST_CASE_1 = [
    {"sigmoid": True, "softmax": False, "other": None},
    torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
    torch.tensor([[[[0.5000, 0.7311], [0.8808, 0.9526]]]]),
    (1, 1, 2, 2),
]

TEST_CASE_2 = [
    {"sigmoid": False, "softmax": True, "other": None},
    torch.tensor([[[[0.0, 1.0]], [[2.0, 3.0]]]]),
    torch.tensor([[[[0.1192, 0.1192]], [[0.8808, 0.8808]]]]),
    (1, 2, 1, 2),
]

TEST_CASE_3 = [
    {"sigmoid": False, "softmax": False, "other": lambda x: torch.tanh(x)},
    torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
    torch.tensor([[[[0.0000, 0.7616], [0.9640, 0.9951]]]]),
    (1, 1, 2, 2),
]

TEST_CASE_4 = [
    "swish",
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    torch.tensor(
        [[[[-4.54e-04, -2.68e-03, -1.48e-02, -7.19e-02, -2.38e-01], [0.00e00, 1.76e00, 3.93e00, 5.99e00, 8.00e00]]]]
    ),
    (1, 1, 2, 5),
]

TEST_CASE_5 = [
    "mish",
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    torch.tensor(
        [[[[-4.54e-04, -2.68e-03, -1.49e-02, -7.26e-02, -2.53e-01], [0.00e00, 1.94e00, 4.00e00, 6.00e00, 8.00e00]]]]
    ),
    (1, 1, 2, 5),
]


class TestActivations(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value_shape(self, input_param, img, out, expected_shape):
        result = Activations(**input_param)(img)
        torch.testing.assert_allclose(result, out)
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
    def test_monai_activations_value_shape(self, input_param, img, out, expected_shape):
        act = Act[input_param]()
        result = act(img)
        torch.testing.assert_allclose(result, out, rtol=1e-2, atol=1e-5)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
