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
            for dtype in (np.float32, np.float64):
                for align in (False, True):
                    for interp_mode in ("nearest", "bilinear"):
                        for padding_mode in ("zeros", "border", "reflection"):
                            TESTS.append(
                                [
                                    {},  # default no params
                                    np.arange(4).reshape((1, 2, 2)) + 1.0,  # data
                                    {
                                        "src": p_src(np.eye(3)),
                                        "dst": p(dst) if dst is not None else None,
                                        "dtype": dtype,
                                        "align_corners": align,
                                        "mode": interp_mode,
                                        "padding_mode": padding_mode,
                                    },
                                    np.array([[[2.0, 1.0], [4.0, 3.0]]])
                                    if ind == 0
                                    else np.array([[[3.0, 4.0], [1.0, 2.0]]]),
                                ]
                            )

for ind, dst in enumerate(
    [
        np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        np.asarray([[-1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ]
):
    for p in TEST_NDARRAYS:
        for p_src in TEST_NDARRAYS:
            for dtype in (np.float32, np.float64):
                for align in (True, False):
                    interp = ("nearest", "bilinear")
                    if align and USE_COMPILED:
                        interp = interp + (0, 1)  # type: ignore
                    for interp_mode in interp:
                        for padding_mode in ("zeros", "border", "reflection"):
                            TESTS.append(
                                [
                                    {},  # default no params
                                    np.arange(12).reshape((1, 2, 2, 3)) + 1.0,  # data
                                    {
                                        "src": p_src(np.eye(4)),
                                        "dst": p(dst) if dst is not None else None,
                                        "dtype": dtype,
                                        "align_corners": align,
                                        "mode": interp_mode,
                                        "padding_mode": padding_mode,
                                    },
                                    np.array(
                                        [[[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], [[10.0, 11.0, 12.0], [7.0, 8.0, 9.0]]]]
                                    )
                                    if ind == 0
                                    else np.array(
                                        [[[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]
                                    ),
                                ]
                            )


class TestSpatialResample(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_flips(self, init_param, img, data_param, expected_output):
        for p in TEST_NDARRAYS:
            _img = p(img)
            _expected_output = p(expected_output)
            output_data, output_dst = SpatialResample(**init_param)(img=_img, **data_param)
            assert_allclose(output_data, _expected_output)
            expected_dst = data_param.get("dst") if data_param.get("dst") is not None else data_param.get("src")
            assert_allclose(output_dst, expected_dst, type_test=False)


if __name__ == "__main__":
    unittest.main()
