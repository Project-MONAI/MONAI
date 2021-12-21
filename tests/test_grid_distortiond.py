# Copyright (c) MONAI Consortium
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
from parameterized import parameterized

from monai.transforms import GridDistortiond
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
num_cells = (2, 2)
distort_steps = [(1.5,) * (1 + n_c) for n_c in num_cells]
for p in TEST_NDARRAYS:
    img = np.indices([6, 6]).astype(np.float32)
    TESTS.append(
        [
            dict(
                keys=["img", "mask"],
                num_cells=num_cells,
                distort_steps=distort_steps,
                mode=["bilinear", "nearest"],
                padding_mode=["reflection", "zeros"],
            ),
            {"img": p(img), "mask": p(np.ones_like(img[:1]))},
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [2.25, 2.25, 2.25, 2.25, 2.25, 2.25],
                            [4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                            [4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                            [3.25, 3.25, 3.25, 3.25, 3.25, 3.25],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        ],
                        [
                            [0.0, 2.25, 4.5, 4.5, 3.25, 1.0],
                            [0.0, 2.25, 4.5, 4.5, 3.25, 1.0],
                            [0.0, 2.25, 4.5, 4.5, 3.25, 1.0],
                            [0.0, 2.25, 4.5, 4.5, 3.25, 1.0],
                            [0.0, 2.25, 4.5, 4.5, 3.25, 1.0],
                            [0.0, 2.25, 4.5, 4.5, 3.25, 1.0],
                        ],
                    ]
                ).astype(np.float32)
            ),
            p(
                np.array(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ).astype(np.float32)
            ),
        ]
    )


class TestGridDistortiond(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_grid_distortiond(self, input_param, input_data, expected_val_img, expected_val_mask):
        g = GridDistortiond(**input_param)
        result = g(input_data)
        assert_allclose(result["img"], expected_val_img, rtol=1e-4, atol=1e-4)
        assert_allclose(result["mask"], expected_val_mask, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
