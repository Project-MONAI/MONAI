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

from monai.transforms import RandCoarseDropout
from monai.utils import fall_back_tuple
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"holes": 2, "spatial_size": [2, 2, 2], "fill_value": 5, "prob": 1.0},
            p(np.random.randint(0, 2, size=[3, 3, 3, 4])),
            (3, 3, 3, 4),
        ]
    )

    TESTS.append(
        [
            {"holes": 1, "spatial_size": [1, 2, 3], "fill_value": 5, "max_holes": 5, "prob": 1.0},
            p(np.random.randint(0, 2, size=[3, 3, 3, 4])),
            (3, 3, 3, 4),
        ]
    )

    TESTS.append(
        [
            {"holes": 2, "spatial_size": [2, 2, 2], "fill_value": 5, "max_spatial_size": [4, 4, 3], "prob": 1.0},
            p(np.random.randint(0, 2, size=[3, 3, 3, 4])),
            (3, 3, 3, 4),
        ]
    )

    TESTS.append(
        [
            {"holes": 2, "spatial_size": [2, -1, 2], "fill_value": 5, "max_spatial_size": [4, 4, -1], "prob": 1.0},
            p(np.random.randint(0, 2, size=[3, 3, 3, 4])),
            (3, 3, 3, 4),
        ]
    )


class TestRandCoarseDropout(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_param, input_data, expected_shape):
        dropout = RandCoarseDropout(**input_param)
        result = dropout(input_data)
        np.testing.assert_equal(result.shape, expected_shape)
        holes = input_param.get("holes")
        max_holes = input_param.get("max_holes")
        spatial_size = fall_back_tuple(input_param.get("spatial_size"), input_data.shape[1:])
        max_spatial_size = fall_back_tuple(input_param.get("max_spatial_size"), input_data.shape[1:])

        if max_holes is None:
            self.assertEqual(len(dropout.hole_coords), holes)
        else:
            self.assertGreaterEqual(len(dropout.hole_coords), holes)
            self.assertLessEqual(len(dropout.hole_coords), max_holes)

        for h in dropout.hole_coords:
            data = result[h]
            self.assertEqual(type(data), type(input_data))
            if isinstance(data, torch.Tensor):
                self.assertEqual(data.device, input_data.device)
                data = data.cpu()
            np.testing.assert_allclose(data, input_param.get("fill_value", 0))
            if max_spatial_size is None:
                self.assertTupleEqual(data.shape[1:], tuple(spatial_size))
            else:
                for d, s, m in zip(data.shape[1:], spatial_size, max_spatial_size):
                    self.assertGreaterEqual(d, s)
                    self.assertLessEqual(d, m)


if __name__ == "__main__":
    unittest.main()
