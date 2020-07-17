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

from monai.transforms import Rand3DElasticd

TEST_CASES = [
    [
        {
            "keys": ("img", "seg"),
            "magnitude_range": (0.3, 2.3),
            "sigma_range": (1.0, 20.0),
            "prob": 0.0,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, 2, 2),
        },
        {"img": torch.ones((2, 3, 3, 3)), "seg": torch.ones((2, 3, 3, 3))},
        np.ones((2, 2, 2, 2)),
    ],
    [
        {
            "keys": ("img", "seg"),
            "magnitude_range": (0.3, 2.3),
            "sigma_range": (1.0, 20.0),
            "prob": 0.0,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, -1, -1),
        },
        {"img": torch.ones((2, 3, 3, 3)), "seg": torch.ones((2, 3, 3, 3))},
        np.ones((2, 2, 3, 3)),
    ],
    [
        {
            "keys": ("img", "seg"),
            "magnitude_range": (0.3, 2.3),
            "sigma_range": (1.0, 20.0),
            "prob": 0.0,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": -1,
        },
        {"img": torch.arange(8).reshape((1, 2, 2, 2)), "seg": torch.arange(8).reshape((1, 2, 2, 2))},
        np.arange(8).reshape((1, 2, 2, 2)),
    ],
    [
        {
            "keys": ("img", "seg"),
            "magnitude_range": (0.3, 0.3),
            "sigma_range": (1.0, 2.0),
            "prob": 0.9,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, 2, 2),
        },
        {"img": torch.arange(27).reshape((1, 3, 3, 3)), "seg": torch.arange(27).reshape((1, 3, 3, 3))},
        np.array([[[[6.492354, 7.5022864], [9.519528, 10.524366]], [[15.51277, 16.525297], [18.533852, 19.539217]]]]),
    ],
    [
        {
            "keys": ("img", "seg"),
            "magnitude_range": (0.3, 0.3),
            "sigma_range": (1.0, 2.0),
            "prob": 0.9,
            "rotate_range": [1, 1, 1],
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, 2, 2),
            "mode": "bilinear",
        },
        {"img": torch.arange(27).reshape((1, 3, 3, 3)), "seg": torch.arange(27).reshape((1, 3, 3, 3))},
        np.array([[[[5.005563, 9.463698], [9.289501, 13.741863]], [[12.320587, 16.779654], [16.597677, 21.049414]]]]),
    ],
    [
        {
            "keys": ("img", "seg"),
            "mode": ("bilinear", "nearest"),
            "magnitude_range": (0.3, 0.3),
            "sigma_range": (1.0, 2.0),
            "prob": 0.9,
            "rotate_range": [1, 1, 1],
            "as_tensor_output": True,
            "device": torch.device("cpu:0"),
            "spatial_size": (2, 2, 2),
        },
        {"img": torch.arange(27).reshape((1, 3, 3, 3)), "seg": torch.arange(27).reshape((1, 3, 3, 3))},
        {
            "img": torch.tensor([[[[5.0056, 9.4637], [9.2895, 13.7419]], [[12.3206, 16.7797], [16.5977, 21.0494]]]]),
            "seg": torch.tensor([[[[4.0, 14.0], [7.0, 14.0]], [[9.0, 19.0], [12.0, 22.0]]]]),
        },
    ],
]


class TestRand3DElasticd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_3d_elasticd(self, input_param, input_data, expected_val):
        g = Rand3DElasticd(**input_param)
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
