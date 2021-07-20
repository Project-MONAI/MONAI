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

from tests.utils import TEST_NDARRAYS
import unittest
import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import AsDiscreted

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([
        {
            "keys": ["pred", "label"],
            "argmax": [True, False],
            "to_onehot": True,
            "n_classes": 2,
            "threshold_values": False,
            "logit_thresh": 0.5,
        },
        {"pred": p(torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]])), "label": p(torch.tensor([[[0, 1]]]))},
        {"pred": torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]]]), "label": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])},
        (2, 1, 2),
    ])
    TESTS.append([
        {
            "keys": ["pred", "label"],
            "argmax": False,
            "to_onehot": False,
            "n_classes": None,
            "threshold_values": [True, False],
            "logit_thresh": 0.6,
        },
        {"pred": p(torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])), "label": p(torch.tensor([[[0, 1], [1, 1]]]))},
        {"pred": torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]), "label": torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])},
        (1, 2, 2),
    ])
    TESTS.append([
        {
            "keys": ["pred"],
            "argmax": True,
            "to_onehot": True,
            "n_classes": 2,
            "threshold_values": False,
            "logit_thresh": 0.5,
        },
        {"pred": p(torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]]))},
        {"pred": torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]]])},
        (2, 1, 2),
    ])


class TestAsDiscreted(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value_shape(self, input_param, test_input, output, expected_shape):
        result = AsDiscreted(**input_param)(test_input)
        for k in ("label", "pred"):
            if k not in result:
                continue
            i, r, o = test_input[k], result[k], output[k]
            self.assertEqual(type(i), type(r))
            if isinstance(r, torch.Tensor):
                self.assertEqual(i.device, r.device)
                r = r.cpu()
            np.testing.assert_allclose(r, o)


if __name__ == "__main__":
    unittest.main()
