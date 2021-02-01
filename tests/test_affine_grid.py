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

from monai.transforms import AffineGrid

TEST_CASES = [
    [
        {"as_tensor_output": False, "device": torch.device("cpu:0")},
        {"spatial_size": (2, 2)},
        np.array([[[-0.5, -0.5], [0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]], [[1.0, 1.0], [1.0, 1.0]]]),
    ],
    [
        {"as_tensor_output": True, "device": None},
        {"spatial_size": (2, 2)},
        torch.tensor([[[-0.5, -0.5], [0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]], [[1.0, 1.0], [1.0, 1.0]]]),
    ],
    [{"as_tensor_output": False, "device": None}, {"grid": np.ones((3, 3, 3))}, np.ones((3, 3, 3))],
    [{"as_tensor_output": True, "device": torch.device("cpu:0")}, {"grid": np.ones((3, 3, 3))}, torch.ones((3, 3, 3))],
    [{"as_tensor_output": False, "device": None}, {"grid": torch.ones((3, 3, 3))}, np.ones((3, 3, 3))],
    [
        {"as_tensor_output": True, "device": torch.device("cpu:0")},
        {"grid": torch.ones((3, 3, 3))},
        torch.ones((3, 3, 3)),
    ],
    [
        {
            "rotate_params": (1.0, 1.0),
            "scale_params": (-20, 10),
            "as_tensor_output": True,
            "device": torch.device("cpu:0"),
        },
        {"grid": torch.ones((3, 3, 3))},
        torch.tensor(
            [
                [[-19.2208, -19.2208, -19.2208], [-19.2208, -19.2208, -19.2208], [-19.2208, -19.2208, -19.2208]],
                [[-11.4264, -11.4264, -11.4264], [-11.4264, -11.4264, -11.4264], [-11.4264, -11.4264, -11.4264]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ),
    ],
    [
        {
            "rotate_params": (1.0, 1.0, 1.0),
            "scale_params": (-20, 10),
            "as_tensor_output": True,
            "device": torch.device("cpu:0"),
        },
        {"grid": torch.ones((4, 3, 3, 3))},
        torch.tensor(
            [
                [
                    [[-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435]],
                    [[-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435]],
                    [[-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435]],
                ],
                [
                    [[-20.2381, -20.2381, -20.2381], [-20.2381, -20.2381, -20.2381], [-20.2381, -20.2381, -20.2381]],
                    [[-20.2381, -20.2381, -20.2381], [-20.2381, -20.2381, -20.2381], [-20.2381, -20.2381, -20.2381]],
                    [[-20.2381, -20.2381, -20.2381], [-20.2381, -20.2381, -20.2381], [-20.2381, -20.2381, -20.2381]],
                ],
                [
                    [[-0.5844, -0.5844, -0.5844], [-0.5844, -0.5844, -0.5844], [-0.5844, -0.5844, -0.5844]],
                    [[-0.5844, -0.5844, -0.5844], [-0.5844, -0.5844, -0.5844], [-0.5844, -0.5844, -0.5844]],
                    [[-0.5844, -0.5844, -0.5844], [-0.5844, -0.5844, -0.5844], [-0.5844, -0.5844, -0.5844]],
                ],
                [
                    [[1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000]],
                    [[1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000]],
                    [[1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000], [1.0000, 1.0000, 1.0000]],
                ],
            ]
        ),
    ],
]


class TestAffineGrid(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_affine_grid(self, input_param, input_data, expected_val):
        g = AffineGrid(**input_param)
        result = g(**input_data)
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected_val, torch.Tensor))
        if isinstance(result, torch.Tensor):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
