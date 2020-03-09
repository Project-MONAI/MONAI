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

from monai.transforms import RandZoom
from tests.utils import NumpyImageTestCase2D


class ZoomTest(NumpyImageTestCase2D):

    @parameterized.expand([
        (0.9, 1.1, 3, 'constant', 0, True, False, False),
    ])
    def test_correct_results(self, min_zoom, max_zoom, order, mode, 
                             cval, prefilter, use_gpu, keep_size):
        random_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, order=order, 
                               mode=mode, cval=cval, prefilter=prefilter, use_gpu=use_gpu,
                               keep_size=keep_size)
        random_zoom.set_random_state(234)

        zoomed = random_zoom(self.imt)
        expected = zoom_scipy(self.imt, zoom=random_zoom._zoom, mode=mode,
                              order=order, cval=cval, prefilter=prefilter)

        self.assertTrue(np.allclose(expected, zoomed))

    @parameterized.expand([
        (0.8, 1.2, 1, 'constant', 0, True)
    ])
    def test_gpu_zoom(self, min_zoom, max_zoom, order, mode, cval, prefilter):
        if importlib.util.find_spec('cupy'):
            random_zoom = RandZoom(
                prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, order=order, 
                mode=mode, cval=cval, prefilter=prefilter, use_gpu=True, 
                keep_size=False)
            random_zoom.set_random_state(234)

            zoomed = random_zoom(self.imt)
            expected = zoom_scipy(self.imt, zoom=random_zoom._zoom, mode=mode, order=order,
                                  cval=cval, prefilter=prefilter)

            self.assertTrue(np.allclose(expected, zoomed))

    def test_keep_size(self):
        random_zoom = RandZoom(prob=1.0, min_zoom=0.6, 
                               max_zoom=0.7, keep_size=True)
        zoomed = random_zoom(self.imt)
        self.assertTrue(np.array_equal(zoomed.shape, self.imt.shape))

    @parameterized.expand([
        ("no_min_zoom", None, 1.1, 1, TypeError),
        ("invalid_order", 0.9, 1.1 , 's', AssertionError)
    ])
    def test_invalid_inputs(self, _, min_zoom, max_zoom, order, raises):
        with self.assertRaises(raises):
            random_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, order=order)
            zoomed = random_zoom(self.imt)


if __name__ == '__main__':
    unittest.main()
