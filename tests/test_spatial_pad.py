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
from typing import List

import numpy as np
from parameterized import parameterized

from monai.transforms import SpatialPad
from monai.utils.enums import NumpyPadMode
from monai.utils.misc import set_determinism
from tests.utils import TEST_NDARRAYS

TESTS = []

# Numpy modes
MODES: List = [
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]
MODES += [NumpyPadMode(i) for i in MODES]

for mode in MODES:
    TESTS.append(
        [
            {"spatial_size": [50, 50], "method": "end", "mode": mode},
            (1, 2, 2),
            (1, 50, 50),
        ]
    )

    TESTS.append(
        [
            {"spatial_size": [15, 4, -1], "method": "symmetric", "mode": mode},
            (3, 8, 8, 4),
            (3, 15, 8, 4),
        ]
    )


class TestSpatialPad(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    @staticmethod
    def get_arr(shape):
        return np.random.randint(100, size=shape).astype(float)

    @parameterized.expand(TESTS)
    def test_pad_shape(self, input_param, input_shape, expected_shape):
        results_1 = []
        results_2 = []
        input_data = self.get_arr(input_shape)
        for p in TEST_NDARRAYS:
            padder = SpatialPad(**input_param)
            results_1.append(padder(p(input_data)))
            results_2.append(padder(p(input_data), mode=input_param["mode"]))
            for results in (results_1, results_2):
                np.testing.assert_allclose(results[-1].shape, expected_shape)
                if input_param["mode"] not in ("empty", NumpyPadMode.EMPTY):
                    np.testing.assert_allclose(results[0], results[-1], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
