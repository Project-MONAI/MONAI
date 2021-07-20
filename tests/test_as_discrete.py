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

from monai.transforms import AsDiscrete
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"argmax": True, "to_onehot": False, "n_classes": None, "threshold_values": False, "logit_thresh": 0.5},
            p([[[0.0, 1.0]], [[2.0, 3.0]]]),
            ([[[1.0, 1.0]]]),
            (1, 1, 2),
        ]
    )

    TESTS.append(
        [
            {"argmax": True, "to_onehot": True, "n_classes": 2, "threshold_values": False, "logit_thresh": 0.5},
            p([[[0.0, 1.0]], [[2.0, 3.0]]]),
            ([[[0.0, 0.0]], [[1.0, 1.0]]]),
            (2, 1, 2),
        ]
    )

    TESTS.append(
        [
            {"argmax": False, "to_onehot": False, "n_classes": None, "threshold_values": True, "logit_thresh": 0.6},
            p([[[0.0, 1.0], [2.0, 3.0]]]),
            ([[[0.0, 1.0], [1.0, 1.0]]]),
            (1, 2, 2),
        ]
    )

    TESTS.append(
        [
            {"argmax": False, "to_onehot": True, "n_classes": 3},
            p(1),
            ([0.0, 1.0, 0.0]),
            (3,),
        ]
    )


class TestAsDiscrete(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value_shape(self, input_param, img, out, expected_shape):
        result = AsDiscrete(**input_param)(img)
        self.assertEqual(type(img), type(result))
        if isinstance(result, torch.Tensor):
            self.assertEqual(img.device, result.device)
            result = result.cpu()
        np.testing.assert_allclose(result, out)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
