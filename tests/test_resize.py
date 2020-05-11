# Copyright 2020 MONAI Consortium
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
import skimage
from parameterized import parameterized

from monai.transforms import Resize
from tests.utils import NumpyImageTestCase2D


class TestResize(NumpyImageTestCase2D):
    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            resize = Resize(spatial_size=(128, 128, 3), order="order")
            resize(self.imt[0])

    @parameterized.expand(
        [
            ((64, 64), 1, "reflect", 0, True, True, True),
            ((32, 32), 2, "constant", 3, False, False, False),
            ((256, 256), 3, "constant", 3, False, False, False),
        ]
    )
    def test_correct_results(self, spatial_size, order, mode, cval, clip, preserve_range, anti_aliasing):
        resize = Resize(
            spatial_size,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
        )
        expected = list()
        for channel in self.imt[0]:
            expected.append(
                skimage.transform.resize(
                    channel,
                    spatial_size,
                    order=order,
                    mode=mode,
                    cval=cval,
                    clip=clip,
                    preserve_range=preserve_range,
                    anti_aliasing=anti_aliasing,
                )
            )
        expected = np.stack(expected).astype(np.float32)
        self.assertTrue(np.allclose(resize(self.imt[0]), expected))


if __name__ == "__main__":
    unittest.main()
