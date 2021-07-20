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

from monai.transforms import Activationsd
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"keys": ["pred", "label"], "sigmoid": False, "softmax": [True, False], "other": None},
            {
                "pred": p(torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]])),
                "label": p(torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]])),
            },
            {
                "pred": torch.tensor([[[0.1192, 0.1192]], [[0.8808, 0.8808]]]),
                "label": torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]]),
            },
            (2, 1, 2),
        ]
    )
    TESTS.append(
        [
            {"keys": ["pred", "label"], "sigmoid": False, "softmax": False, "other": [torch.tanh, None]},
            {"pred": p(torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])), "label": p(torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]))},
            {
                "pred": torch.tensor([[[0.0000, 0.7616], [0.9640, 0.9951]]]),
                "label": torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
            },
            (1, 2, 2),
        ]
    )
    TESTS.append(
        [
            {"keys": "pred", "sigmoid": False, "softmax": False, "other": torch.tanh},
            {"pred": p(torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]))},
            {"pred": torch.tensor([[[0.0000, 0.7616], [0.9640, 0.9951]]])},
            (1, 2, 2),
        ]
    )


class TestActivationsd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value_shape(self, input_param, test_input, output, expected_shape):
        result = Activationsd(**input_param)(test_input)
        for k in ("label", "pred"):
            if k not in result:
                continue
            i, r, o = test_input[k], result[k], output[k]
            self.assertEqual(type(i), type(r))
            if isinstance(r, torch.Tensor):
                self.assertEqual(r.device, i.device)
                r = r.cpu()
            np.testing.assert_allclose(r, o, rtol=1e-4, atol=1e-5)
            self.assertTupleEqual(r.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
