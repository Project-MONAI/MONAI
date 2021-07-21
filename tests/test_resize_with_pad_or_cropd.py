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

from monai.transforms import ResizeWithPadOrCropd
from tests.utils import TEST_NDARRAYS

TEST_CASES = [
    [
        {"keys": "img", "spatial_size": [15, 8, 8], "mode": "constant"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 8, 8),
    ],
    [
        {"keys": "img", "spatial_size": [15, 4, 8], "mode": "constant", "method": "end", "constant_values": 1},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 8),
    ],
    [
        {"keys": "img", "spatial_size": [15, 4, -1], "mode": "constant"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 4),
    ],
    [
        {"keys": "img", "spatial_size": [15, 4, -1], "mode": "reflect"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 4),
    ],
    [
        {"keys": "img", "spatial_size": [-1, -1, -1], "mode": "reflect"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 8, 8, 4),
    ],
]


class TestResizeWithPadOrCropd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_pad_shape(self, input_param, input_data, expected_shape):
        paddcroper = ResizeWithPadOrCropd(**input_param)
        results = []
        for p in TEST_NDARRAYS:
            input_data_mod = {"img": p(input_data["img"])}
            result = paddcroper(input_data_mod)
            r, i = result["img"], input_data_mod["img"]
            self.assertEqual(type(i), type(r))
            if isinstance(r, torch.Tensor):
                self.assertEqual(r.device, i.device)
                r = r.cpu().numpy()
            np.testing.assert_allclose(r.shape, expected_shape)
            results.append(r)
            # check output from numpy torch and torch.cuda match
            if len(results) > 1:
                np.testing.assert_allclose(results[0], results[-1])


if __name__ == "__main__":
    unittest.main()
