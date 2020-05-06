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

from monai.transforms import Rand3DElastic

TEST_CASES = [
    [
        {
            "magnitude_range": (0.3, 2.3),
            "sigma_range": (1.0, 20.0),
            "prob": 0.0,
            "as_tensor_output": False,
            "device": None,
        },
        {"img": torch.ones((2, 3, 3, 3)), "spatial_size": (2, 2, 2)},
        np.ones((2, 2, 2, 2)),
    ],
    [
        {
            "magnitude_range": (0.3, 0.3),
            "sigma_range": (1.0, 2.0),
            "prob": 0.9,
            "as_tensor_output": False,
            "device": None,
        },
        {"img": torch.arange(27).reshape((1, 3, 3, 3)), "spatial_size": (2, 2, 2)},
        np.array([[[[3.2385552, 4.753422], [7.779232, 9.286472]], [[16.769115, 18.287868], [21.300673, 22.808704]]]]),
    ],
    [
        {
            "magnitude_range": (0.3, 0.3),
            "sigma_range": (1.0, 2.0),
            "prob": 0.9,
            "rotate_range": [1, 1, 1],
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, 2, 2),
        },
        {"img": torch.arange(27).reshape((1, 3, 3, 3)), "mode": "bilinear"},
        np.array([[[[1.6566806, 7.695548], [7.4342523, 13.580086]], [[11.776854, 18.669481], [18.396517, 21.551771]]]]),
    ],
]


class TestRand3DElastic(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_3d_elastic(self, input_param, input_data, expected_val):
        g = Rand3DElastic(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertEqual(torch.is_tensor(result), torch.is_tensor(expected_val))
        if torch.is_tensor(result):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
