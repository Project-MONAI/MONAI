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
from monai.transforms import SpatialResampleD
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
                                "src": p_src(np.eye(3)),
                                "dst": p(dst),
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
                                "src": p_src(np.eye(4)),
                                "dst": p_src(dst),
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
    def test_flips_inverse(self, p_type, args):
        _, img, data_param, expected_output = args
        _img = p_type(img)
        _expected_output = p_type(expected_output)
        input_dict = {"img": _img, "img_meta_dict": {"src": data_param.get("src"), "dst": data_param.get("dst")}}
        xform = SpatialResampleD(
            keys="img",
            meta_src_keys="src",
            meta_dst_keys="dst",
            mode=data_param.get("mode"),
            padding_mode=data_param.get("padding_mode"),
            align_corners=data_param.get("align_corners"),
        )
        output_data = xform(input_dict)
        assert_allclose(output_data["img"], _expected_output, rtol=1e-2, atol=1e-2)
        assert_allclose(
            output_data["img_meta_dict"]["src"], data_param.get("dst"), type_test=False, rtol=1e-2, atol=1e-2
        )

        inverted = xform.inverse(output_data)
        self.assertEqual(inverted["img_transforms"], [])  # no further invert after inverting
        assert_allclose(inverted["img_meta_dict"]["src"], data_param.get("src"), type_test=False, rtol=1e-2, atol=1e-2)
        assert_allclose(inverted["img"], _img, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
