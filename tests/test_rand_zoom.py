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

from monai.transforms import RandZoom
from monai.utils import InterpolateMode
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

VALID_CASES = [(0.8, 1.2, "nearest", False), (0.8, 1.2, InterpolateMode.NEAREST, False)]


class TestRandZoom(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, min_zoom, max_zoom, mode, keep_size):
        for p in TEST_NDARRAYS:
            random_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, mode=mode, keep_size=keep_size)
            random_zoom.set_random_state(1234)
            zoomed = random_zoom(p(self.imt[0]))
            expected = [
                zoom_scipy(channel, zoom=random_zoom._zoom, mode="nearest", order=0, prefilter=False)
                for channel in self.imt[0]
            ]

            expected = np.stack(expected).astype(np.float32)
            assert_allclose(zoomed, p(expected), atol=1.0)

    def test_keep_size(self):
        for p in TEST_NDARRAYS:
            im = p(self.imt[0])
            random_zoom = RandZoom(prob=1.0, min_zoom=0.6, max_zoom=0.7, keep_size=True)
            zoomed = random_zoom(im)
            self.assertTrue(np.array_equal(zoomed.shape, self.imt.shape[1:]))
            zoomed = random_zoom(im)
            self.assertTrue(np.array_equal(zoomed.shape, self.imt.shape[1:]))
            zoomed = random_zoom(im)
            self.assertTrue(np.array_equal(zoomed.shape, self.imt.shape[1:]))

    @parameterized.expand(
        [("no_min_zoom", None, 1.1, "bilinear", TypeError), ("invalid_mode", 0.9, 1.1, "s", ValueError)]
    )
    def test_invalid_inputs(self, _, min_zoom, max_zoom, mode, raises):
        for p in TEST_NDARRAYS:
            with self.assertRaises(raises):
                random_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, mode=mode)
                random_zoom(p(self.imt[0]))

    def test_auto_expand_3d(self):
        for p in TEST_NDARRAYS:
            random_zoom = RandZoom(prob=1.0, min_zoom=[0.8, 0.7], max_zoom=[1.2, 1.3], mode="nearest", keep_size=False)
            random_zoom.set_random_state(1234)
            test_data = p(np.random.randint(0, 2, size=[2, 2, 3, 4]))
            zoomed = random_zoom(test_data)
            assert_allclose(random_zoom._zoom, (1.048844, 1.048844, 0.962637), atol=1e-2)
            assert_allclose(zoomed.shape, (2, 2, 3, 3))


if __name__ == "__main__":
    unittest.main()
