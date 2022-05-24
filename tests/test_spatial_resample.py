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

import itertools
import unittest

import numpy as np
from parameterized import parameterized

from monai.config import USE_COMPILED
from monai.transforms import SpatialResample
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []

for ind, dst in enumerate(
    [
        np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]]),  # flip the second
        np.asarray([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # flip the first
    ]
):
    for p in TEST_NDARRAYS:
        for p_src in TEST_NDARRAYS:
            for align in (False, True):
                for interp_mode in ("nearest", "bilinear"):
                    TESTS.append(
                        [
                            {},  # default no params
                            np.arange(4).reshape((1, 2, 2)) + 1.0,  # data
                            {
                                "src_affine": p_src(np.eye(3)),
                                "dst_affine": p(dst),
                                "dtype": np.float32,
                                "align_corners": align,
                                "mode": interp_mode,
                                "padding_mode": "zeros",
                            },
                            np.array([[[2.0, 1.0], [4.0, 3.0]]]) if ind == 0 else np.array([[[3.0, 4.0], [1.0, 2.0]]]),
                        ]
                    )

for ind, dst in enumerate(
    [
        np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        np.asarray([[-1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ]
):
    for p_src in TEST_NDARRAYS:
        for align in (True, False):
            if align and USE_COMPILED:
                interp = ("nearest", "bilinear", 0, 1)
            else:
                interp = ("nearest", "bilinear")  # type: ignore
            for interp_mode in interp:  # type: ignore
                for padding_mode in ("zeros", "border", "reflection"):
                    TESTS.append(
                        [
                            {},  # default no params
                            np.arange(12).reshape((1, 2, 2, 3)) + 1.0,  # data
                            {
                                "src_affine": p_src(np.eye(4)),
                                "dst_affine": p_src(dst),
                                "dtype": np.float64,
                                "align_corners": align,
                                "mode": interp_mode,
                                "padding_mode": padding_mode,
                            },
                            np.array([[[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], [[10.0, 11.0, 12.0], [7.0, 8.0, 9.0]]]])
                            if ind == 0
                            else np.array(
                                [[[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]
                            ),
                        ]
                    )


class TestSpatialResample(unittest.TestCase):
    @parameterized.expand(itertools.product(TEST_NDARRAYS, TESTS))
    def test_flips(self, p_type, args):
        init_param, img, data_param, expected_output = args
        _img = p_type(img)
        _expected_output = p_type(expected_output)
        output_data, output_dst = SpatialResample(**init_param)(img=_img, **data_param)
        assert_allclose(output_data, _expected_output, rtol=1e-2, atol=1e-2)
        expected_dst = (
            data_param.get("dst_affine") if data_param.get("dst_affine") is not None else data_param.get("src_affine")
        )
        assert_allclose(output_dst, expected_dst, type_test=False, rtol=1e-2, atol=1e-2)

    @parameterized.expand(itertools.product([True, False], TEST_NDARRAYS))
    def test_4d_5d(self, is_5d, p_type):
        new_shape = (1, 2, 2, 3, 1, 1) if is_5d else (1, 2, 2, 3, 1)
        img = np.arange(12).reshape(new_shape)
        img = np.tile(img, (1, 1, 1, 1, 2, 2) if is_5d else (1, 1, 1, 1, 2))
        _img = p_type(img)
        dst = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.5], [0.0, 0.0, 0.0, 1.0]])
        output_data, output_dst = SpatialResample(dtype=np.float32)(
            img=_img, src_affine=p_type(np.eye(4)), dst_affine=dst
        )
        expected_data = (
            np.asarray(
                [
                    [
                        [[[0.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [1.5, 1.0]], [[1.0, 2.0], [2.0, 2.0]]],
                        [[[3.0, 3.0], [3.0, 4.0]], [[3.5, 3.0], [4.5, 4.0]], [[4.0, 5.0], [5.0, 5.0]]],
                    ],
                    [
                        [[[6.0, 6.0], [6.0, 7.0]], [[6.5, 6.0], [7.5, 7.0]], [[7.0, 8.0], [8.0, 8.0]]],
                        [[[9.0, 9.0], [9.0, 10.0]], [[9.5, 9.0], [10.5, 10.0]], [[10.0, 11.0], [11.0, 11.0]]],
                    ],
                ],
                dtype=np.float32,
            )
            if is_5d
            else np.asarray(
                [
                    [[[0.5, 0.0], [0.0, 2.0], [1.5, 1.0]], [[3.5, 3.0], [3.0, 5.0], [4.5, 4.0]]],
                    [[[6.5, 6.0], [6.0, 8.0], [7.5, 7.0]], [[9.5, 9.0], [9.0, 11.0], [10.5, 10.0]]],
                ],
                dtype=np.float32,
            )
        )
        assert_allclose(output_data, p_type(expected_data[None]), rtol=1e-2, atol=1e-2)
        assert_allclose(output_dst, dst, type_test=False, rtol=1e-2, atol=1e-2)

    def test_ill_affine(self):
        img = np.arange(12).reshape(1, 2, 2, 3)
        ill_affine = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        )
        with self.assertRaises(ValueError):
            SpatialResample()(img=img, src_affine=np.eye(4), dst_affine=ill_affine)
        with self.assertRaises(ValueError):
            SpatialResample()(img=img, src_affine=ill_affine, dst_affine=np.eye(3))
        with self.assertRaises(ValueError):
            SpatialResample(mode=None)(img=img, src_affine=np.eye(4), dst_affine=0.1 * np.eye(4))


if __name__ == "__main__":
    unittest.main()
