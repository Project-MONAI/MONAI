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

from monai.transforms import Resize
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS: List[Tuple] = []
for p in TEST_NDARRAYS:
    TESTS.append((p, (32, -1), "area"))
    TESTS.append((p, (32, 32), "area"))
    TESTS.append((p, (32, 32, 32), "trilinear"))
    TESTS.append((p, (256, 256), "bilinear"))


class TestResize(NumpyImageTestCase2D):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resize = Resize(spatial_size=(128, 128, 3), mode="order")
            resize(self.imt[0])

        with self.assertRaises(ValueError):
            resize = Resize(spatial_size=(128,), mode="order")
            resize(self.imt[0])

    @parameterized.expand(TESTS)
    def test_correct_results(self, in_type, spatial_size, mode):
        resize = Resize(spatial_size, mode=mode)
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
        im = in_type(self.imt[0])
        out = resize(im)
        self.assertEqual(type(im), type(out))
        if isinstance(out, torch.Tensor):
            self.assertEqual(im.device, out.device)
            out = out.cpu()
        np.testing.assert_allclose(out, expected, atol=0.9)


if __name__ == "__main__":
    unittest.main()
