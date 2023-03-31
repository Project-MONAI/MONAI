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
import torch
from parameterized import parameterized

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.spatial.functional import flip
from monai.transforms.spatial.array import Flip
from tests.utils import TEST_DEVICES, TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

INVALID_CASES = [("wrong_axis", ["s", 1], TypeError), ("not_numbers", "s", TypeError)]

VALID_CASES = [("no_axis", None), ("one_axis", 1), ("many_axis", [0, 1]), ("negative_axis", [0, -1])]

TORCH_CASES = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TORCH_CASES.append([[0, 1], torch.zeros((1, 3, 2)), track_meta, *device])


class TestFlip(NumpyImageTestCase2D):
    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, _, spatial_axis, raises):
        with self.assertRaises(raises):
            flip = Flip(spatial_axis)
            flip(self.imt[0])

    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, spatial_axis):
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            flip = Flip(spatial_axis=spatial_axis)
            expected = [np.flip(channel, spatial_axis) for channel in self.imt[0]]
            expected = np.stack(expected)
            result = flip(im)
            assert_allclose(result, p(expected), type_test="tensor")
            test_local_inversion(flip, result, im)

    @parameterized.expand(TORCH_CASES)
    def test_torch(self, init_param, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        img = img.to(device)
        xform = Flip(init_param)
        res = xform(img)
        self.assertEqual(img.shape, res.shape)
        if track_meta:
            self.assertIsInstance(res, MetaTensor)
        else:
            self.assertNotIsInstance(res, MetaTensor)
            self.assertIsInstance(res, torch.Tensor)
            with self.assertRaisesRegex(ValueError, "MetaTensor"):
                xform.inverse(res)


def prep_affine(dims, modifications=None):
    a = torch.eye(dims, dtype=torch.float64)
    if modifications is not None:
        for m in modifications:
            a[m[0], m[1]] = m[2]
    return a


FUNCTIONAL_TORCH_CASES = [
    (tuple(), True, tuple(), prep_affine(4)),
    ((0,), True, (0,), prep_affine(4, ((0, 0, -1),))),
    ((1,), True, (1,), prep_affine(4, ((1, 1, -1),))),
    ((2,), True, (2,), prep_affine(4, ((2, 2, -1),))),
    ((0, 1), True, (0, 1), prep_affine(4, ((0, 0, -1), (1, 1, -1)))),
    ((0, 2), True, (0, 2), prep_affine(4, ((0, 0, -1), (2, 2, -1)))),
    ((1, 2), True, (1, 2), prep_affine(4, ((1, 1, -1), (2, 2, -1)))),
    ((0, 1, 2), True, (0, 1, 2), prep_affine(4, ((0, 0, -1), (1, 1, -1), (2, 2, -1)))),
    (None, True, (0, 1, 2), prep_affine(4, ((0, 0, -1), (1, 1, -1), (2, 2, -1)))),
]


class TestFlipFunction(unittest.TestCase):

    @parameterized.expand(FUNCTIONAL_TORCH_CASES)
    def test_flip(self, spatial_axes, lazy, expected_spatial_axis, expected_affine):
        data = torch.zeros((1, 32, 32, 16))
        result = flip(data, spatial_axis=spatial_axes, lazy= lazy)
        pending = result.pending_operations[-1]
        # print(result.shape, pending)
        self.assertTupleEqual(pending.metadata['spatial_axis'], expected_spatial_axis)
        self.assertTrue(torch.allclose(pending.matrix.data, expected_affine))


if __name__ == "__main__":
    unittest.main()
