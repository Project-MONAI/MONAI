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

from monai.transforms import Rand3DElastic

TEST_CASES = [
    [
        {
            "magnitude_range": (0.3, 2.3),
            "sigma_range": (1.0, 20.0),
            "prob": 0.0,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": -1,
        },
        {"img": torch.arange(72).reshape((2, 3, 3, 4))},
        np.arange(72).reshape((2, 3, 3, 4)),
    ],
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
        np.array([[[[6.4939356, 7.50289], [9.518351, 10.522849]], [[15.512375, 16.523542], [18.531467, 19.53646]]]]),
    ],
    [
        {
            "magnitude_range": (0.3, 0.3),
            "sigma_range": (1.0, 2.0),
            "prob": 0.9,
            "rotate_range": [1, 1, 1],
            "as_tensor_output": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "spatial_size": (2, 2, 2),
        },
        {"img": torch.arange(27).reshape((1, 3, 3, 3)), "mode": "bilinear"},
        np.array([[[[5.0069294, 9.463932], [9.287769, 13.739735]], [[12.319424, 16.777205], [16.594296, 21.045748]]]]),
    ],
    [
        {
            "magnitude_range": (0.3, 0.3),
            "sigma_range": (1.0, 2.0),
            "prob": 0.9,
            "translate_range": [0.1, 0.1, 0.1],
            "translate_percent": True,
            "as_tensor_output": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "spatial_size": (2, 2, 2),
        },
        {"img": torch.arange(27).reshape((1, 3, 3, 3)), "mode": "bilinear"},
        np.array([[[[7.119943, 8.128898], [10.144359, 11.148856]], [[16.138384, 17.14955], [19.157475, 20.16247]]]]),
    ],
]


class TestRand3DElastic(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_3d_elastic(self, input_param, input_data, expected_val):
        g = Rand3DElastic(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected_val, torch.Tensor))
        if isinstance(result, torch.Tensor):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
