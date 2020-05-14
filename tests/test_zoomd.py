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
import importlib

from scipy.ndimage import zoom as zoom_scipy
from parameterized import parameterized

from monai.transforms import Zoomd
from tests.utils import NumpyImageTestCase2D

VALID_CASES = [
    (1.1, 3, "constant", 0, True, False, False),
    (0.9, 3, "constant", 0, True, False, False),
    (0.8, 1, "reflect", 0, False, False, False),
]

GPU_CASES = [("gpu_zoom", 0.6, 1, "constant", 0, True)]

INVALID_CASES = [("no_zoom", None, 1, TypeError), ("invalid_order", 0.9, "s", TypeError)]


class TestZoomd(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, zoom, order, mode, cval, prefilter, use_gpu, keep_size):
        key = "img"
        zoom_fn = Zoomd(
            key, zoom=zoom, order=order, mode=mode, cval=cval, prefilter=prefilter, use_gpu=use_gpu, keep_size=keep_size
        )
        zoomed = zoom_fn({key: self.imt[0]})
        expected = list()
        for channel in self.imt[0]:
            expected.append(zoom_scipy(channel, zoom=zoom, mode=mode, order=order, cval=cval, prefilter=prefilter))
        expected = np.stack(expected).astype(np.float32)
        self.assertTrue(np.allclose(expected, zoomed[key]))

    @parameterized.expand(GPU_CASES)
    def test_gpu_zoom(self, _, zoom, order, mode, cval, prefilter):
        key = "img"
        if importlib.util.find_spec("cupy"):
            zoom_fn = Zoomd(
                key, zoom=zoom, order=order, mode=mode, cval=cval, prefilter=prefilter, use_gpu=True, keep_size=False
            )
            zoomed = zoom_fn({key: self.imt[0]})
            expected = list()
            for channel in self.imt[0]:
                expected.append(zoom_scipy(channel, zoom=zoom, mode=mode, order=order, cval=cval, prefilter=prefilter))
            expected = np.stack(expected).astype(np.float32)
            self.assertTrue(np.allclose(expected, zoomed[key]))

    def test_keep_size(self):
        key = "img"
        zoom_fn = Zoomd(key, zoom=0.6, keep_size=True)
        zoomed = zoom_fn({key: self.imt[0]})
        self.assertTrue(np.array_equal(zoomed[key].shape, self.imt.shape[1:]))

        zoom_fn = Zoomd(key, zoom=1.3, keep_size=True)
        zoomed = zoom_fn({key: self.imt[0]})
        self.assertTrue(np.array_equal(zoomed[key].shape, self.imt.shape[1:]))

    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, _, zoom, order, raises):
        key = "img"
        with self.assertRaises(raises):
            zoom_fn = Zoomd(key, zoom=zoom, order=order)
            zoomed = zoom_fn({key: self.imt[0]})


if __name__ == "__main__":
    unittest.main()
