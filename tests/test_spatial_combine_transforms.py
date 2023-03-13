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

from monai.data.meta_tensor import MetaTensor
from monai.transforms import (
    Orientation,
    Orientationd,
    RandFlip,
    RandFlipd,
    Randomizable,
    RandRotate,
    RandRotated,
    RandSpatialCrop,
    RandZoom,
    RandZoomd,
    Resize,
    Resized,
    Spacing,
    Spacingd,
)
from monai.transforms.transform import MapTransform
from tests.utils import assert_allclose

TEST_2D = [
    [
        (2, 64, 64),
        [
            (Spacing, {"pixdim": (1.2, 1.5), "padding_mode": "zeros", "dtype": torch.float32}),
            (Orientation, {"axcodes": "RA"}),
            (Resize, {"spatial_size": (48, 48), "mode": "bilinear"}),
            (RandSpatialCrop, {"roi_size": (32, 32)}),
            # TODO: the following transform should also work?
            # (RandAffine, {"prob": 0.9, "rotate_range": (np.pi / 2,), "shear_range": [1, 2], "translate_range": [2, 1], "mode": "bilinear"}),
            (RandFlip, {"prob": 0.9}),
            (RandRotate, {"prob": 0.9, "range_x": np.pi / 4}),
            (RandZoom, {"prob": 0.9, "mode": "bilinear", "keep_size": False, "align_corners": False}),
        ],
    ],
    [
        (2, 64, 64),
        [
            # TODO: the following transform should also work?
            # (RandSpatialCropd, {"roi_size": (32, 32), "keys": "img"}),
            (Spacingd, {"pixdim": (1.2, 1.5), "padding_mode": "zeros", "dtype": torch.float32, "keys": "img"}),
            (Orientationd, {"axcodes": "RA", "keys": "img"}),
            (Resized, {"spatial_size": (48, 48), "mode": "bilinear", "keys": "img"}),
            # TODO: the following transform should also work?
            # (RandAffined, {"prob": 0.9, "rotate_range": (np.pi / 2,), "shear_range": [1, 2], "translate_range": [2, 1], "mode": "bilinear", "keys": "img"}),
            (RandFlipd, {"prob": 0.9, "keys": "img"}),
            (RandRotated, {"prob": 0.9, "range_x": np.pi / 4, "keys": "img"}),
            (RandZoomd, {"prob": 0.9, "mode": "bilinear", "keep_size": False, "keys": "img", "align_corners": False}),
        ],
    ],
]

TEST_3D = [
    [
        (2, 64, 64, 64),
        [
            (Spacing, {"pixdim": (1.2, 1.5, 0.9), "padding_mode": "zeros", "dtype": torch.float32}),
            (Orientation, {"axcodes": "RAS"}),
            (Resize, {"spatial_size": (32, 32, 32), "mode": "nearest"}),
            (RandFlip, {"prob": 0.9}),
            (RandRotate, {"prob": 0.9, "range_x": np.pi / 4}),
            (RandZoom, {"prob": 0.9, "mode": "nearest", "keep_size": False}),
        ],
    ],
    [
        (2, 64, 64, 64),
        [
            (Spacingd, {"pixdim": (1.2, 1.5, 0.9), "padding_mode": "zeros", "dtype": torch.float32, "keys": "img"}),
            (Orientationd, {"axcodes": "RAS", "keys": "img"}),
            (Resized, {"spatial_size": (32, 32, 32), "mode": "nearest", "keys": "img"}),
            (RandFlipd, {"prob": 0.9, "keys": "img"}),
            (RandRotated, {"prob": 0.9, "range_x": np.pi / 4, "keys": "img"}),
            (RandZoomd, {"prob": 0.9, "mode": "nearest", "keep_size": False, "keys": "img"}),
        ],
    ],
]


class CombineLazyTest(unittest.TestCase):
    @parameterized.expand(TEST_2D + TEST_3D)
    def test_combine_transforms(self, input_shape, funcs):
        for seed in [10, 100, 1000, 10000]:
            _funcs = []
            for _func, _params in funcs:
                _funcs.append(_func(**_params))
            is_map = isinstance(_funcs[0], MapTransform)
            data = torch.randint(low=1, high=10, size=input_shape).float()
            im = MetaTensor(data, meta={"a": "b", "affine": np.eye(len(input_shape))})
            input_data = {"img": im} if is_map else im
            # non lazy
            non_lazy_result = input_data
            for _func in _funcs:
                if isinstance(_func, Randomizable):
                    _func.set_random_state(seed=seed)
                non_lazy_result = _func(non_lazy_result)
            expected = non_lazy_result["img"] if is_map else non_lazy_result

            # lazy
            pending_result = input_data
            for _func in _funcs:
                _func.lazy_evaluation = True
                if isinstance(_func, Randomizable):
                    _func.set_random_state(seed=seed)
                pending_result = _func(pending_result)
            pending_result = pending_result["img"] if is_map else pending_result

            assert_allclose(pending_result.peek_pending_affine(), expected.affine)
            assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:4])
            # # TODO: how to test final result?
            # init_param = funcs[-1][1]
            # call_param = {}
            # apply_param = get_apply_param(init_param, call_param)
            # result = apply_transforms(pending_result, **apply_param)[0]
            # assert_allclose(result, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
