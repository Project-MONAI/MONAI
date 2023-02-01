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

from __future__ import annotations

import unittest

import numpy as np
import skimage.transform
import torch
from parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Resize
from monai.transforms.spatial.functional import resize
from tests.utils import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, is_tf32_env, pytorch_after

TEST_CASE_0 = [{"spatial_size": 15}, (6, 10, 15)]

TEST_CASE_1 = [{"spatial_size": 15, "mode": "area"}, (6, 10, 15)]

TEST_CASE_2 = [{"spatial_size": 6, "mode": "trilinear", "align_corners": True}, (2, 4, 6)]

TEST_CASE_3 = [{"spatial_size": 15, "anti_aliasing": True}, (6, 10, 15)]

TEST_CASE_4 = [{"spatial_size": 6, "anti_aliasing": True, "anti_aliasing_sigma": 2.0}, (2, 4, 6)]

TEST_CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4]

ext_keys = ['spatial_size', 'size_mode', 'mode', 'align_corners', 'anti_aliasing',
            'anti_aliasing_sigma', 'dtype', 'shape_override']

TEST_CASES_EXT = [
    [(1, 32, 32), {"spatial_size": 16},
     dict(zip(ext_keys, [16, 'all', 'area', False, None, None, torch.float64, (1, 16, 16)]))],
    [(1, 32, 32), {"spatial_size": 15, 'mode': 'nearest'},
     dict(zip(ext_keys, [15, 'all', 'nearest', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": 15, 'mode': 'nearest-exact'},
     dict(zip(ext_keys, [15, 'all', 'nearest-exact', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": 15, 'mode': 'linear'},
     dict(zip(ext_keys, [15, 'all', 'linear', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": 15, 'mode': 'bilinear'},
     dict(zip(ext_keys, [15, 'all', 'bilinear', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": 15, 'mode': 'bicubic'},
     dict(zip(ext_keys, [15, 'all', 'bicubic', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": 15, 'mode': 'trilinear'},
     dict(zip(ext_keys, [15, 'all', 'trilinear', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": 15, "mode": "area"},
     dict(zip(ext_keys, [15, 'all', 'area', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": (15, 15), "mode": "area"},
     dict(zip(ext_keys, [(15, 15), 'all', 'area', False, None, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32, 16), {"spatial_size": (15, 15, 12), "mode": "bilinear"},
     dict(zip(ext_keys, [(15, 15, 12), 'all', 'bilinear', False, None, None, torch.float64, (1, 15, 15, 12)]))],
    [(1, 32, 32), {"spatial_size": 6, "mode": "trilinear", "align_corners": True},
     dict(zip(ext_keys, [6, 'all', 'trilinear', True, None, None, torch.float64, (1, 6, 6)]))],
    [(1, 32, 32), {"spatial_size": 15, "anti_aliasing": True},
     dict(zip(ext_keys, [15, 'all', 'area', False, True, None, torch.float64, (1, 15, 15)]))],
    [(1, 32, 32), {"spatial_size": 6, "anti_aliasing": True, "anti_aliasing_sigma": 2.0},
     dict(zip(ext_keys, [6, 'all', 'area', False, True, 2.0, torch.float64, (1, 6, 6)]))]
]

diff_t = 0.3 if is_tf32_env() else 0.2

CORRECT_RESULTS_TESTS = [
    # TODO: reinstate area tests once the resampler allows it
    ((32, -1), "bilinear", True), # ((32, -1), "area", True),
    ((32, 32), "bilinear", False), # ((32, 32), "area", False),
    ((32, 32, 32), "trilinear", True),
    ((256, 256), "bilinear", False),
    ((256, 256), "nearest-exact" if pytorch_after(1, 11) else "nearest", False),
    ((128, 64), "bilinear", True), # ((128, 64), "area", True),  # already in a good shape
]


class TestResize(NumpyImageTestCase2D):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resize = Resize(spatial_size=(128, 128, 3), mode="order")
            resize(self.imt[0])

        with self.assertRaises(ValueError):
            resize = Resize(spatial_size=(128,), mode="order")
            resize(self.imt[0])

    @parameterized.expand(TEST_CASES_EXT)
    def test_functional_resize(self, img_size, kwargs, expected):
        self._test_functional_resize(img_size, kwargs, expected)

    def test_functional_resize_cases(self):
        for t in TEST_CASES_EXT:
            with self.subTest(t):
                self._test_functional_resize(t[0], t[1], t[2])

    def test_function_resize_not_lazy_cases(self):
        for t in TEST_CASES_EXT:
            with self.subTest(t):
                self._test_functional_resize(t[0], t[1], t[2], False)

    def _test_functional_resize(self, img_size, kwargs, expected, lazy: bool = True):
        print(kwargs)
        img = np.random.rand(*img_size)
        result = resize(img, **kwargs, lazy_evaluation=lazy)
        if lazy is True:
            self.assertDictEqual(result.pending_operations[-1].metadata, expected)

        else:
            print(result.applied_operations, result.affine)
            result.applied_operations


    @parameterized.expand(CORRECT_RESULTS_TESTS)
    def test_correct_results(self, spatial_size, mode, anti_aliasing):
        self._test_correct_results(spatial_size, mode, anti_aliasing)

    def test_correct_results_cases(self):
        for t in CORRECT_RESULTS_TESTS:
            with self.subTest(t):
                self._test_correct_results(*t)

    def _test_correct_results(self, spatial_size, mode, anti_aliasing):
        """resize 'spatial_size' and 'mode'"""
        resize = Resize(spatial_size, mode=mode, anti_aliasing=anti_aliasing)
        _order = 0
        if mode.endswith("linear"):
            _order = 1
        if spatial_size == (32, -1):
            spatial_size = (32, 64)

        expected = [
            skimage.transform.resize(
                channel, spatial_size, order=_order, clip=False, preserve_range=False, anti_aliasing=anti_aliasing
            )
            for channel in self.imt[0]
        ]

        expected = np.stack(expected).astype(np.float32)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            out = resize(im)
            if isinstance(im, MetaTensor):
                im_inv = resize.inverse(out)
                self.assertTrue(not im_inv.applied_operations)
                assert_allclose(im_inv.shape, im.shape)
                assert_allclose(im_inv.affine, im.affine, atol=1e-3, rtol=1e-3)
            if not anti_aliasing:
                assert_allclose(out, expected, type_test=False, atol=0.9)
                return
            # skimage uses reflect padding for anti-aliasing filter.
            # Our implementation reuses GaussianSmooth() as anti-aliasing filter, which uses zero padding instead.
            # Thus their results near the image boundary will be different.
            if isinstance(out, torch.Tensor):
                out = out.cpu().detach().numpy()
            good = np.sum(np.isclose(expected, out, atol=0.9))
            self.assertLessEqual(
                np.abs(good - expected.size) / float(expected.size), diff_t, f"at most {diff_t} percent mismatch "
            )

    @parameterized.expand(TEST_CASES)
    def test_longest_shape(self, input_param, expected_shape):
        self._test_longest_shape(input_param, expected_shape)

    def test_longest_shape_cases(self):
        for t in TEST_CASES:
            with self.subTest(t):
                self._test_longest_shape(*t)

    def _test_longest_shape(self, input_param, expected_shape):
        input_data = np.random.randint(0, 2, size=[3, 4, 7, 10]).astype(np.double)
        input_param["size_mode"] = "longest"
        result = Resize(**input_param, lazy_evaluation=False)(input_data)
        np.testing.assert_allclose(result.shape[1:], expected_shape)

        set_track_meta(False)
        result = Resize(**input_param, lazy_evaluation=False)(input_data)
        self.assertNotIsInstance(result, MetaTensor)
        np.testing.assert_allclose(result.shape[1:], expected_shape)
        set_track_meta(True)

    def test_longest_infinite_decimals(self):
        resize = Resize(spatial_size=1008, size_mode="longest", mode="bilinear", align_corners=False)
        ret = resize(np.random.randint(0, 2, size=[1, 2544, 3032]))
        self.assertTupleEqual(ret.shape, (1, 846, 1008))


if __name__ == "__main__":
    unittest.main()
