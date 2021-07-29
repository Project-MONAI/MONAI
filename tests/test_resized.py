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
from typing import List, Tuple

import numpy as np
import skimage.transform
import torch
from parameterized import parameterized

from monai.transforms import Resized
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS: List[Tuple] = []
TEST_LONGEST: List[Tuple] = []
for p in TEST_NDARRAYS:
    TESTS.append((p, (32, -1), "area"))
    TESTS.append((p, (64, 64), "area"))
    TESTS.append((p, (32, 32, 32), "area"))
    TESTS.append((p, (256, 256), "bilinear"))

    TEST_LONGEST.append((p, {"keys": "img", "spatial_size": 15}, (6, 11, 15)))
    TEST_LONGEST.append((p, {"keys": "img", "spatial_size": 15, "mode": "area"}, (6, 11, 15)))
    TEST_LONGEST.append((p, {"keys": "img", "spatial_size": 6, "mode": "trilinear", "align_corners": True}, (3, 5, 6)))
    TEST_LONGEST.append(
        (
            p,
            {
                "keys": ["img", "label"],
                "spatial_size": 6,
                "mode": ["trilinear", "nearest"],
                "align_corners": [True, None],
            },
            (3, 5, 6),
        )
    )


class TestResized(NumpyImageTestCase2D):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resize = Resized(keys="img", spatial_size=(128, 128, 3), mode="order")
            resize({"img": self.imt[0]})

        with self.assertRaises(ValueError):
            resize = Resized(keys="img", spatial_size=(128,), mode="order")
            resize({"img": self.imt[0]})

    @parameterized.expand(TESTS)
    def test_correct_results(self, in_type, spatial_size, mode):
        resize = Resized("img", spatial_size, mode=mode)
        _order = 0
        if mode.endswith("linear"):
            _order = 1
        if spatial_size == (32, -1):
            spatial_size = (32, 64)
        expected = []
        for channel in self.imt[0]:
            expected.append(
                skimage.transform.resize(
                    channel, spatial_size, order=_order, clip=False, preserve_range=False, anti_aliasing=False
                )
            )
        expected = np.stack(expected).astype(np.float32)
        out = resize({"img": in_type(self.imt[0])})["img"]
        if isinstance(out, torch.Tensor):
            out = out.cpu()
        np.testing.assert_allclose(out, expected, atol=0.9)

    @parameterized.expand(TEST_LONGEST)
    def test_longest_shape(self, in_type, input_param, expected_shape):
        input_data = {
            "img": in_type(np.random.randint(0, 2, size=[3, 4, 7, 10])),
            "label": in_type(np.random.randint(0, 2, size=[3, 4, 7, 10])),
        }
        input_param["size_mode"] = "longest"
        rescaler = Resized(**input_param)
        result = rescaler(input_data)
        for k in rescaler.keys:
            np.testing.assert_allclose(result[k].shape[1:], expected_shape)


if __name__ == "__main__":
    unittest.main()
