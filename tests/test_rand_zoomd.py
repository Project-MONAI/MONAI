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

from monai.data import MetaTensor
from monai.transforms import RandZoomd
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

VALID_CASES = [(0.8, 1.2, "nearest", None, False)]


class TestRandZoomd(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, min_zoom, max_zoom, mode, align_corners, keep_size):
        key = "img"
        random_zoom = RandZoomd(
            key,
            prob=1.0,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            mode=mode,
            align_corners=align_corners,
            keep_size=keep_size,
        )
        for p in TEST_NDARRAYS:
            random_zoom.set_random_state(1234)

            im = p(self.imt[0])
            zoomed = random_zoom({key: im})
            if isinstance(im, MetaTensor):
                im_inv = random_zoom.inverse(zoomed)
                self.assertTrue(not im_inv["img"].applied_operations)
                assert_allclose(im_inv["img"].shape, im.shape)
                assert_allclose(im_inv["img"].affine, im.affine, atol=1e-3, rtol=1e-3)
            expected = [
                zoom_scipy(channel, zoom=random_zoom.rand_zoom._zoom, mode="nearest", order=0, prefilter=False)
                for channel in self.imt[0]
            ]

            expected = np.stack(expected).astype(np.float32)
            assert_allclose(zoomed[key], p(expected), atol=1.0, type_test=False)

    def test_keep_size(self):
        key = "img"
        random_zoom = RandZoomd(
            keys=key, prob=1.0, min_zoom=0.6, max_zoom=0.7, keep_size=True, padding_mode="constant", constant_values=2
        )
        for p in TEST_NDARRAYS:
            im = p(self.imt[0])
            zoomed = random_zoom({key: im})
            if isinstance(im, MetaTensor):
                im_inv = random_zoom.inverse(zoomed)
                self.assertTrue(not im_inv["img"].applied_operations)
                assert_allclose(im_inv["img"].shape, im.shape)
                assert_allclose(im_inv["img"].affine, im.affine, atol=1e-3, rtol=1e-3)
            np.testing.assert_array_equal(zoomed[key].shape, self.imt.shape[1:])

    @parameterized.expand(
        [("no_min_zoom", None, 1.1, "bilinear", TypeError), ("invalid_order", 0.9, 1.1, "s", ValueError)]
    )
    def test_invalid_inputs(self, _, min_zoom, max_zoom, mode, raises):
        key = "img"
        for p in TEST_NDARRAYS:
            with self.assertRaises(raises):
                random_zoom = RandZoomd(key, prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, mode=mode)
                random_zoom({key: p(self.imt[0])})

    def test_auto_expand_3d(self):
        random_zoom = RandZoomd(
            keys="img", prob=1.0, min_zoom=[0.8, 0.7], max_zoom=[1.2, 1.3], mode="nearest", keep_size=False
        )
        for p in TEST_NDARRAYS:
            random_zoom.set_random_state(1234)
            test_data = {"img": p(np.random.randint(0, 2, size=[2, 2, 3, 4]))}
            zoomed = random_zoom(test_data)
            assert_allclose(random_zoom.rand_zoom._zoom, (1.048844, 1.048844, 0.962637), atol=1e-2)
            assert_allclose(zoomed["img"].shape, (2, 2, 3, 3))


if __name__ == "__main__":
    unittest.main()
