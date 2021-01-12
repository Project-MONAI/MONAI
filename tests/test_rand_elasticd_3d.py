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
        np.array([[[[6.4939356, 7.50289], [9.518351, 10.522849]], [[15.512375, 16.523542], [18.531467, 19.53646]]]]),
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
        np.array([[[[5.0069294, 9.463932], [9.287769, 13.739735]], [[12.319424, 16.777205], [16.594296, 21.045748]]]]),
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
            "img": torch.tensor([[[[5.0069, 9.4639], [9.2878, 13.7397]], [[12.3194, 16.7772], [16.5943, 21.0457]]]]),
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
