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
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [
                {"device": device},
                {"spatial_size": (2, 2)},
                np.array([[[-0.5, -0.5], [0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]], [[1.0, 1.0], [1.0, 1.0]]]),
            ]
        )

        TESTS.append([{"device": device}, {"grid": p(np.ones((3, 3, 3)))}, p(np.ones((3, 3, 3)))])
        TESTS.append([{"device": device}, {"grid": p(torch.ones((3, 3, 3)))}, p(np.ones((3, 3, 3)))])
        TESTS.append(
            [
                {
                    "rotate_params": (1.0, 1.0),
                    "scale_params": (-20, 10),
                    "device": device,
                },
                {"grid": p(torch.ones((3, 3, 3)))},
                p(
                    torch.tensor(
                        [
                            [
                                [-19.2208, -19.2208, -19.2208],
                                [-19.2208, -19.2208, -19.2208],
                                [-19.2208, -19.2208, -19.2208],
                            ],
                            [
                                [-11.4264, -11.4264, -11.4264],
                                [-11.4264, -11.4264, -11.4264],
                                [-11.4264, -11.4264, -11.4264],
                            ],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        ]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                {
                    "rotate_params": (1.0, 1.0, 1.0),
                    "scale_params": (-20, 10),
                    "device": device,
                },
                {"grid": p(torch.ones((4, 3, 3, 3)))},
                p(
                    torch.tensor(
                        [
                            [
                                [[-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435]],
                                [[-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435]],
                                [[-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435], [-9.5435, -9.5435, -9.5435]],
                            ],
                            [
                                [
                                    [-20.2381, -20.2381, -20.2381],
                                    [-20.2381, -20.2381, -20.2381],
                                    [-20.2381, -20.2381, -20.2381],
                                ],
                                [
                                    [-20.2381, -20.2381, -20.2381],
                                    [-20.2381, -20.2381, -20.2381],
                                    [-20.2381, -20.2381, -20.2381],
                                ],
                                [
                                    [-20.2381, -20.2381, -20.2381],
                                    [-20.2381, -20.2381, -20.2381],
                                    [-20.2381, -20.2381, -20.2381],
                                ],
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
                    )
                ),
            ]
        )


class TestAffineGrid(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_affine_grid(self, input_param, input_data, expected_val):
        g = AffineGrid(**input_param)
        result, _ = g(**input_data)
        if "grid" in input_data:
            self.assertEqual(type(result), type(expected_val))
            if isinstance(result, torch.Tensor):
                self.assertEqual(result.device, expected_val.device)
                result = result.cpu()
                expected_val = expected_val.cpu()
        np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
