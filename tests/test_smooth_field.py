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

from monai.transforms import RandSmoothFieldAdjustContrastd, RandSmoothFieldAdjustIntensityd
from tests.utils import TEST_NDARRAYS, assert_allclose, is_tf32_env

_rtol = 5e-3 if is_tf32_env() else 1e-4

INPUT_SHAPE1 = (1, 8, 8)
INPUT_SHAPE2 = (2, 8, 8)

TESTS_CONTRAST = []
TESTS_INTENSITY = []

for p in TEST_NDARRAYS:
    TESTS_CONTRAST += [
        (
            {"keys": ("test",), "spatial_size": INPUT_SHAPE1[1:], "rand_size": (4, 4), "prob": 1.0},
            {"test": p(np.ones(INPUT_SHAPE1, np.float32))},
            {"test": p(np.ones(INPUT_SHAPE1, np.float32))},
        ),
        (
            {"keys": ("test",), "spatial_size": INPUT_SHAPE2[1:], "rand_size": (4, 4), "prob": 1.0},
            {"test": p(np.ones(INPUT_SHAPE2, np.float32))},
            {"test": p(np.ones(INPUT_SHAPE2, np.float32))},
        ),
    ]

    TESTS_INTENSITY += [
        (
            {"keys": ("test",), "spatial_size": INPUT_SHAPE1[1:], "rand_size": (4, 4), "prob": 1.0, "gamma": (1, 1)},
            {"test": p(np.ones(INPUT_SHAPE1, np.float32))},
            {"test": p(np.ones(INPUT_SHAPE1, np.float32))},
        ),
        (
            {"keys": ("test",), "spatial_size": INPUT_SHAPE2[1:], "rand_size": (4, 4), "prob": 1.0, "gamma": (1, 1)},
            {"test": p(np.ones(INPUT_SHAPE2, np.float32))},
            {"test": p(np.ones(INPUT_SHAPE2, np.float32))},
        ),
    ]


class TestSmoothField(unittest.TestCase):
    @parameterized.expand(TESTS_CONTRAST)
    def test_rand_smooth_field_adjust_contrastd(self, input_param, input_data, expected_val):
        g = RandSmoothFieldAdjustContrastd(**input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=5e-3)

    @parameterized.expand(TESTS_INTENSITY)
    def test_rand_smooth_field_adjust_intensityd(self, input_param, input_data, expected_val):
        g = RandSmoothFieldAdjustIntensityd(**input_param)
        g.set_random_state(123)

        res = g(input_data)
        for key, result in res.items():
            expected = expected_val[key]
            assert_allclose(result, expected, rtol=_rtol, atol=5e-3)
