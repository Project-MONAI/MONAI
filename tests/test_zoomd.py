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
from scipy.ndimage import zoom as zoom_scipy

from monai.transforms import Zoomd
from tests.utils import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

VALID_CASES = [(1.5, "nearest", False), (0.3, "bilinear", False), (0.8, "bilinear", False)]

INVALID_CASES = [("no_zoom", None, "bilinear", TypeError), ("invalid_order", 0.9, "s", ValueError)]


class TestZoomd(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, zoom, mode, keep_size):
        key = "img"
        zoom_fn = Zoomd(key, zoom=zoom, mode=mode, keep_size=keep_size)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            zoomed = zoom_fn({key: im})
            test_local_inversion(zoom_fn, zoomed, {key: im}, key)
            _order = 0
            if mode.endswith("linear"):
                _order = 1
            expected = [
                zoom_scipy(channel, zoom=zoom, mode="nearest", order=_order, prefilter=False) for channel in self.imt[0]
            ]

            expected = np.stack(expected).astype(np.float32)
            assert_allclose(zoomed[key], p(expected), atol=1.0, type_test=False)

    def test_keep_size(self):
        key = "img"
        zoom_fn = Zoomd(key, zoom=0.6, keep_size=True, padding_mode="constant", constant_values=2)
        for p in TEST_NDARRAYS_ALL:
            zoomed = zoom_fn({key: p(self.imt[0])})
            np.testing.assert_array_equal(zoomed[key].shape, self.imt.shape[1:])

            zoom_fn = Zoomd(key, zoom=1.3, keep_size=True)
            zoomed = zoom_fn({key: self.imt[0]})
            self.assertTrue(np.array_equal(zoomed[key].shape, self.imt.shape[1:]))

    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, _, zoom, mode, raises):
        key = "img"
        for p in TEST_NDARRAYS_ALL:
            with self.assertRaises(raises):
                zoom_fn = Zoomd(key, zoom=zoom, mode=mode)
                zoom_fn({key: p(self.imt[0])})


if __name__ == "__main__":
    unittest.main()
