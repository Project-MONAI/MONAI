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
            "device": None,
            "spatial_size": (2, 2),
            "mode": "bilinear",
        },
        {"img": torch.ones((3, 3, 3)), "seg": torch.ones((3, 3, 3))},
        np.array([[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]]),
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
                [[1.2427862, 2.9714181], [4.0735664, 5.35417]],
                [[10.242786, 11.971418], [13.073566, 14.35417]],
                [[19.242786, 20.971418], [22.073566, 23.35417]],
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
                    [[1.2428, 2.9714], [4.0736, 5.3542]],
                    [[10.2428, 11.9714], [13.0736, 14.3542]],
                    [[19.2428, 20.9714], [22.0736, 23.3542]],
                ]
            ),
            "seg": torch.tensor([[[0.0, 2.0], [3.0, 4.0]], [[9.0, 11.0], [12.0, 13.0]], [[18.0, 20.0], [21.0, 22.0]]]),
        },
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
            self.assertEqual(torch.is_tensor(result), torch.is_tensor(expected))
            if torch.is_tensor(result):
                np.testing.assert_allclose(result.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-4)
            else:
                np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
