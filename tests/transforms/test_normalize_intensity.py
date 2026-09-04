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

from monai.data import MetaTensor, get_track_meta, set_track_meta
from monai.transforms import Compose, NormalizeIntensity
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p, {"nonzero": True}, np.array([0.0, 3.0, 0.0, 4.0]), np.array([0.0, -1.0, 0.0, 1.0])])
    for q in TEST_NDARRAYS:
        for u in TEST_NDARRAYS:
            TESTS.append(
                [
                    p,
                    {
                        "subtrahend": q(np.array([3.5, 3.5, 3.5, 3.5])),
                        "divisor": u(np.array([0.5, 0.5, 0.5, 0.5])),
                        "nonzero": True,
                    },
                    p(np.array([0.0, 3.0, 0.0, 4.0])),
                    p(np.array([0.0, -1.0, 0.0, 1.0])),
                ]
            )
    TESTS.append([p, {"nonzero": True}, p(np.array([0.0, 0.0, 0.0, 0.0])), p(np.array([0.0, 0.0, 0.0, 0.0]))])
    TESTS.append([p, {"nonzero": False}, p(np.array([0.0, 0.0, 0.0, 0.0])), p(np.array([0.0, 0.0, 0.0, 0.0]))])
    TESTS.append([p, {"nonzero": False}, p(np.array([1, 1, 1, 1])), p(np.array([0.0, 0.0, 0.0, 0.0]))])
    TESTS.append(
        [
            p,
            {"nonzero": False, "channel_wise": True, "subtrahend": [1, 2, 3], "dtype": np.float32},
            p(np.ones((3, 2, 2))),
            p(np.array([[[0.0, 0.0], [0.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]], [[-2.0, -2.0], [-2.0, -2.0]]])),
        ]
    )
    TESTS.append(
        [
            p,
            {"nonzero": True, "channel_wise": True, "subtrahend": [1, 2, 3], "divisor": [0, 0, 2], "dtype": "float32"},
            p(np.ones((3, 2, 2))),
            p(np.array([[[0.0, 0.0], [0.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]], [[-1.0, -1.0], [-1.0, -1.0]]])),
        ]
    )
    TESTS.append(
        [
            p,
            {"nonzero": True, "channel_wise": False, "subtrahend": 2, "divisor": 0, "dtype": torch.float32},
            p(np.ones((3, 2, 2))),
            p(np.ones((3, 2, 2)) * -1.0),
        ]
    )
    TESTS.append(
        [
            p,
            {"nonzero": True, "channel_wise": False, "subtrahend": np.ones((3, 2, 2)) * 0.5, "divisor": 0},
            p(np.ones((3, 2, 2))),
            p(np.ones((3, 2, 2)) * 0.5),
        ]
    )
    TESTS.append(
        [
            p,
            {"nonzero": True, "channel_wise": True, "subtrahend": np.ones((3, 2, 2)) * 0.5, "divisor": [0, 1, 0]},
            p(np.ones((3, 2, 2))),
            p(np.ones((3, 2, 2)) * 0.5),
        ]
    )


class TestNormalizeIntensity(NumpyImageTestCase2D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_default(self, im_type):
        im = im_type(self.imt.copy())
        normalizer = NormalizeIntensity()
        normalized = normalizer(im)
        self.assertTrue(normalized.dtype in (np.float32, torch.float32))
        expected = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        assert_allclose(normalized, expected, type_test="tensor", rtol=1e-3)

    @parameterized.expand(TESTS)
    def test_nonzero(self, in_type, input_param, input_data, expected_data):
        normalizer = NormalizeIntensity(**input_param)
        im = in_type(input_data)
        normalized = normalizer(im)
        assert_allclose(normalized, in_type(expected_data), type_test="tensor")

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise(self, im_type):
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)
        input_data = im_type(np.array([[0.0, 3.0, 0.0, 4.0], [0.0, 4.0, 0.0, 5.0]]))
        expected = np.array([[0.0, -1.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0]])
        normalized = normalizer(input_data)
        assert_allclose(normalized, im_type(expected), type_test="tensor")

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise_int(self, im_type):
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)
        input_data = im_type(torch.arange(1, 25).reshape(2, 3, 4))
        expected = np.array(
            [
                [
                    [-1.593255, -1.3035723, -1.0138896, -0.7242068],
                    [-0.4345241, -0.1448414, 0.1448414, 0.4345241],
                    [0.7242068, 1.0138896, 1.3035723, 1.593255],
                ],
                [
                    [-1.593255, -1.3035723, -1.0138896, -0.7242068],
                    [-0.4345241, -0.1448414, 0.1448414, 0.4345241],
                    [0.7242068, 1.0138896, 1.3035723, 1.593255],
                ],
            ]
        )
        normalized = normalizer(input_data)
        assert_allclose(normalized, im_type(expected), type_test="tensor", rtol=1e-7, atol=1e-7)  # tolerance

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_value_errors(self, im_type):
        input_data = im_type(np.array([[0.0, 3.0, 0.0, 4.0], [0.0, 4.0, 0.0, 5.0]]))
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True, subtrahend=[1])
        with self.assertRaises(ValueError):
            normalizer(input_data)
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True, subtrahend=[1, 2], divisor=[1])
        with self.assertRaises(ValueError):
            normalizer(input_data)

    @parameterized.expand(
        [
            ["global_computed", {}],
            ["channelwise_computed", {"channel_wise": True}],
            ["global_explicit", {"subtrahend": 2.0, "divisor": 3.0}],
            ["channelwise_explicit", {"subtrahend": [1.0, 2.0, 3.0], "divisor": [2.0, 3.0, 4.0], "channel_wise": True}],
            ["nonzero", {"nonzero": True}],
            ["channelwise_nonzero", {"nonzero": True, "channel_wise": True}],
            ["nonzero_explicit", {"nonzero": True, "subtrahend": 2.0, "divisor": 3.0}],
        ]
    )
    def test_inverse(self, _, args):
        self.addCleanup(set_track_meta, get_track_meta())
        set_track_meta(True)
        img = MetaTensor(torch.randn(3, 6, 6) * 5 + 2)
        img[0, :2] = 0  # some zero voxels, which nonzero=True must leave untouched
        img[2] = 0  # an all-zero channel, where nonzero=True has nothing to normalize
        normalizer = NormalizeIntensity(**args)
        out = normalizer(img.clone())
        inv = normalizer.inverse(out)
        assert_allclose(inv, img, type_test=False, rtol=1e-4, atol=1e-4)
        self.assertEqual(len(inv.applied_operations), 0)

    @parameterized.expand([["global", {}], ["channelwise", {"channel_wise": True}]])
    def test_inverse_nonzero_value_equal_to_mean(self, _, args):
        """A non-zero voxel equal to the mean becomes exactly 0 and must still be restored."""
        self.addCleanup(set_track_meta, get_track_meta())
        set_track_meta(True)
        # mean of the non-zero voxels is 2 globally and in each channel, so the voxels equal to 2 become 0
        img = MetaTensor(torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 2.0]]))
        normalizer = NormalizeIntensity(nonzero=True, **args)
        out = normalizer(img.clone())
        self.assertEqual(out[0, 2].item(), 0.0)
        self.assertEqual(out[1, 3].item(), 0.0)
        inv = normalizer.inverse(out)
        assert_allclose(inv, img, type_test=False, rtol=0, atol=0)

    def test_inverse_nonzero_in_compose(self):
        self.addCleanup(set_track_meta, get_track_meta())
        set_track_meta(True)
        img = MetaTensor(torch.randn(2, 5, 5))
        img[0, 0] = 0
        transform = Compose([NormalizeIntensity(nonzero=True)])
        inv = transform.inverse(transform(img.clone()))
        assert_allclose(inv, img, type_test=False, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
