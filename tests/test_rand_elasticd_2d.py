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

from monai.transforms import Rand2DElasticd

TEST_CASES = [
    [
        {
            "keys": ("img", "seg"),
            "spacing": (0.3, 0.3),
            "magnitude_range": (1.0, 2.0),
            "prob": 0.0,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.ones((3, 3, 3)), "seg": torch.ones((3, 3, 3))},
        np.ones((3, 2, 2)),
    ],
    [
        {
            "keys": ("img", "seg"),
            "spacing": (0.3, 0.3),
            "magnitude_range": (0.3, 0.3),
            "prob": 0.0,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": -1,
        },
        {"img": torch.arange(4).reshape((1, 2, 2)), "seg": torch.arange(4).reshape((1, 2, 2))},
        np.arange(4).reshape((1, 2, 2)),
    ],
    [
        {
            "keys": ("img", "seg"),
            "spacing": (0.3, 0.3),
            "magnitude_range": (1.0, 2.0),
            "prob": 0.9,
            "as_tensor_output": False,
            "padding_mode": "zeros",
            "device": None,
            "spatial_size": (2, 2),
            "mode": "bilinear",
        },
        {"img": torch.ones((3, 3, 3)), "seg": torch.ones((3, 3, 3))},
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
            "keys": ("img", "seg"),
            "spacing": (1.0, 1.0),
            "magnitude_range": (1.0, 1.0),
            "scale_range": [1.2, 2.2],
            "prob": 0.9,
            "padding_mode": "border",
            "as_tensor_output": True,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3)), "seg": torch.arange(27).reshape((3, 3, 3))},
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
            "keys": ("img", "seg"),
            "spacing": (0.3, 0.3),
            "magnitude_range": (0.1, 0.2),
            "translate_range": [-0.01, 0.01],
            "scale_range": [0.01, 0.02],
            "prob": 0.9,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3)), "seg": torch.arange(27).reshape((3, 3, 3))},
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
            "keys": ("img", "seg"),
            "mode": ("bilinear", "nearest"),
            "spacing": (0.3, 0.3),
            "magnitude_range": (0.1, 0.2),
            "translate_range": [-0.01, 0.01],
            "scale_range": [0.01, 0.02],
            "prob": 0.9,
            "as_tensor_output": True,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3)), "seg": torch.arange(27).reshape((3, 3, 3))},
        {
            "img": torch.tensor(
                [
                    [[1.3584, 1.9251], [5.6266, 6.6427]],
                    [[10.3584, 10.9251], [14.6266, 15.6427]],
                    [[19.3584, 19.9251], [23.6266, 24.6427]],
                ]
            ),
            "seg": torch.tensor([[[0.0, 2.0], [6.0, 8.0]], [[9.0, 11.0], [15.0, 17.0]], [[18.0, 20.0], [24.0, 26.0]]]),
        },
    ],
    [
        {
            "keys": "img",
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
        {
            "img": torch.tensor(
                [
                    [[ 2.2469,  0.8962], [ 5.0849,  6.9136]],
                    [[11.2469,  9.8962], [14.0849, 15.9136]],
                    [[20.2469, 18.8962], [23.0849, 24.9136]],
                ]
            ),
        }
    ],
]


class TestRand2DElasticd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_2d_elasticd(self, input_param, input_data, expected_val):
        g = Rand2DElasticd(**input_param)
        g.set_random_state(123)
        res = g(input_data)
        for key in res:
            result = res[key]
            expected = expected_val[key] if isinstance(expected_val, dict) else expected_val
            self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected, torch.Tensor))
            if isinstance(result, torch.Tensor):
                np.testing.assert_allclose(result.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-4)
            else:
                np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
