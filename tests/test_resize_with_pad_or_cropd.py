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
import torch
from parameterized import parameterized

from monai.transforms import ResizeWithPadOrCropd
from tests.utils import TEST_NDARRAYS

TEST_CASES = [
    [{"keys": "img", "spatial_size": [15, 8, 8], "mode": "constant"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 15, 8, 8)],
    [
        {"keys": "img", "spatial_size": [15, 4, 8], "mode": "constant", "method": "end", "constant_values": 1},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 8),
    ],
    [{"keys": "img", "spatial_size": [15, 4, -1], "mode": "constant"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 15, 4, 4)],
    [{"keys": "img", "spatial_size": [15, 4, -1], "mode": "reflect"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 15, 4, 4)],
    [{"keys": "img", "spatial_size": [-1, -1, -1], "mode": "reflect"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 8, 8, 4)],
]


class TestResizeWithPadOrCropd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_pad_shape(self, input_param, input_data, expected_val):
        for p in TEST_NDARRAYS:
            if isinstance(p(0), torch.Tensor) and (
                "constant_values" in input_param or input_param["mode"] == "reflect"
            ):
                continue
            padcropper = ResizeWithPadOrCropd(**input_param)
            input_data["img"] = p(input_data["img"])
            result = padcropper(input_data)
            np.testing.assert_allclose(result["img"].shape, expected_val)
            inv = padcropper.inverse(result)
            for k in input_data:
                self.assertTupleEqual(inv[k].shape, input_data[k].shape)


if __name__ == "__main__":
    unittest.main()
