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

from monai.transforms import Rand2DElastic

TEST_CASES = [
    [
        {"spacing": (0.3, 0.3), "magnitude_range": (1.0, 2.0), "prob": 0.0, "as_tensor_output": False, "device": None},
        {"img": torch.ones((3, 3, 3)), "spatial_size": (2, 2)},
        np.ones((3, 2, 2)),
    ],
    [
        {"spacing": (0.3, 0.3), "magnitude_range": (1.0, 2.0), "prob": 0.0, "as_tensor_output": False, "device": None},
        {"img": torch.arange(27).reshape((3, 3, 3))},
        np.arange(27).reshape((3, 3, 3)),
    ],
    [
        {
            "spacing": (0.3, 0.3),
            "magnitude_range": (1.0, 2.0),
            "prob": 0.9,
            "as_tensor_output": False,
            "device": None,
            "padding_mode": "zeros",
        },
        {"img": torch.ones((3, 3, 3)), "spatial_size": (2, 2), "mode": "bilinear"},
        np.array(
            [
                [[0.45531988, 0.0], [0.0, 0.71558857]],
                [[0.45531988, 0.0], [0.0, 0.71558857]],
                [[0.45531988, 0.0], [0.0, 0.71558857]],
            ]
        ),
    ],
    [
        {
            "spacing": (1.0, 1.0),
            "magnitude_range": (1.0, 1.0),
            "scale_range": [1.2, 2.2],
            "prob": 0.9,
            "padding_mode": "border",
            "as_tensor_output": True,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3))},
        torch.tensor(
            [
                [[3.0793, 2.6141], [4.0568, 5.9978]],
                [[12.0793, 11.6141], [13.0568, 14.9978]],
                [[21.0793, 20.6141], [22.0568, 23.9978]],
            ]
        ),
    ],
    [
        {
            "spacing": (0.3, 0.3),
            "magnitude_range": (0.1, 0.2),
            "translate_range": [-0.01, 0.01],
            "scale_range": [0.01, 0.02],
            "prob": 0.9,
            "as_tensor_output": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3))},
        np.array(
            [
                [[1.3584113, 1.9251312], [5.626623, 6.642721]],
                [[10.358411, 10.925131], [14.626623, 15.642721]],
                [[19.358412, 19.92513], [23.626623, 24.642721]],
            ]
        ),
    ],
    [
        {
            "spacing": (1.0, 1.0),
            "magnitude_range": (1.0, 1.0),
            "translate_range": [0.2, 0.5],
            "translate_percent": True,
            "prob": 1.0,
            "padding_mode": "border",
            "as_tensor_output": True,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3))},
        torch.tensor(
            [
                [[2.2469, 0.8962], [5.0849, 6.9136]],
                [[11.2469, 9.8962], [14.0849, 15.9136]],
                [[20.2469, 18.8962], [23.0849, 24.9136]],
            ]
        ),
    ],
]


class TestRand2DElastic(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_2d_elastic(self, input_param, input_data, expected_val):
        g = Rand2DElastic(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected_val, torch.Tensor))
        if isinstance(result, torch.Tensor):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
