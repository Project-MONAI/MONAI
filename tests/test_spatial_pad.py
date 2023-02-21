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
from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms import SpatialPad
from monai.transforms.lazy.functional import apply_transforms
from tests.padders import PadTest
from tests.utils import assert_allclose

TESTS = []
TESTS.append([{"spatial_size": [3, 4], "method": "end"}, (1, 2, 3), (1, 3, 4)])
TESTS.append([{"spatial_size": [15, 4, -1], "method": "symmetric"}, (3, 8, 8, 4), (3, 15, 8, 4)])
TESTS_PENDING_MODE = [["constant", "zeros"], ["edge", "border"], ["reflect", "reflection"]]


class TestSpatialPad(PadTest):
    Padder = SpatialPad

    @parameterized.expand(TESTS)
    def test_pad(self, input_param, input_shape, expected_shape):
        self.pad_test(input_param, input_shape, expected_shape)

    def test_pad_kwargs(self):
        kwargs = {"spatial_size": [15, 8], "method": "end", "mode": "constant"}
        unchanged_slices = [slice(None), slice(None, 8), slice(None, 4)]
        self.pad_test_kwargs(unchanged_slices, **kwargs)

    @parameterized.expand(TESTS)
    def test_pending_ops(self, input_param, input_shape, _):
        # TODO: One of the dim in the input data contains 1 report error.
        data = np.random.randint(100, size=input_shape).astype(np.float64)
        im = MetaTensor(data, meta={"a": "b", "affine": np.eye(len(input_shape))})

        for mode in TESTS_PENDING_MODE:
            pad_fn = SpatialPad(mode=mode[0], **input_param)
            # non-lazy
            expected = pad_fn(im)
            self.assertIsInstance(expected, MetaTensor)
            # lazy
            pad_fn.lazy_evaluation = True
            pending_result = pad_fn(im)
            self.assertIsInstance(pending_result, MetaTensor)
            assert_allclose(pending_result.peek_pending_affine(), expected.affine)
            assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
            # TODO: mode="bilinear" may report error
            result = apply_transforms(pending_result, mode="nearest", padding_mode=mode[1], align_corners=True)[0]
            # compare
            assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
