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
from parameterized import parameterized

from monai.transforms import RandGridDistortiond
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    num_cells = 2
    seed = 0
    img = np.indices([6, 6]).astype(np.float32)
    TESTS.append(
        [
            dict(
                keys=["img", "mask"],
                num_cells=num_cells,
                prob=1.0,
                spatial_dims=2,
                distort_limit=(-0.1, 0.1),
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
            ),
            seed,
            {"img": p(img), "mask": p(np.ones_like(img[:1]))},
            p(
                np.array(
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.6390989, 1.6390989, 1.6390989, 1.6390989, 1.6390989, 0.0],
                            [3.2781978, 3.2781978, 3.2781978, 3.2781978, 3.2781978, 0.0],
                            [3.2781978, 3.2781978, 3.2781978, 3.2781978, 3.2781978, 0.0],
                            [4.74323, 4.74323, 4.74323, 4.74323, 4.74323, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 1.5086684, 3.0173368, 3.0173368, 4.5377502, 0.0],
                            [0.0, 1.5086684, 3.0173368, 3.0173368, 4.5377502, 0.0],
                            [0.0, 1.5086684, 3.0173368, 3.0173368, 4.5377502, 0.0],
                            [0.0, 1.5086684, 3.0173368, 3.0173368, 4.5377502, 0.0],
                            [0.0, 1.5086684, 3.0173368, 3.0173368, 4.5377502, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ).astype(np.float32)
            ),
            p(
                np.array(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        ]
                    ]
                )
            ),
        ]
    )


class TestRandGridDistortiond(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_grid_distortiond(self, input_param, seed, input_data, expected_val_img, expected_val_mask):
        g = RandGridDistortiond(**input_param)
        g.set_random_state(seed=seed)
        result = g(input_data)
        assert_allclose(result["img"], expected_val_img, rtol=1e-4, atol=1e-4)
        assert_allclose(result["mask"], expected_val_mask, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
