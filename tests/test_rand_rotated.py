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
from typing import List, Tuple

import numpy as np
import scipy.ndimage
import torch
from parameterized import parameterized

from monai.transforms import RandRotated
from monai.utils import GridSampleMode, GridSamplePadMode
from tests.utils import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, NumpyImageTestCase3D, test_local_inversion

TEST_CASES_2D: List[Tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_2D.append((p, np.pi / 2, True, "bilinear", "border", False))
    TEST_CASES_2D.append((p, np.pi / 4, True, "nearest", "border", False))
    TEST_CASES_2D.append((p, np.pi, False, "nearest", "zeros", True))
    TEST_CASES_2D.append((p, (-np.pi / 4, 0), False, "nearest", "zeros", True))


TEST_CASES_3D: List[Tuple] = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES_3D.append(
        (p, np.pi / 2, -np.pi / 6, (0.0, np.pi), False, "bilinear", "border", False, (1, 87, 104, 109))
    )
    TEST_CASES_3D.append(
        (
            p,
            np.pi / 2,
            -np.pi / 6,
            (0.0, np.pi),
            False,
            GridSampleMode.NEAREST,
            GridSamplePadMode.BORDER,
            False,
            (1, 87, 104, 109),
        )
    )
    TEST_CASES_3D.append(
        (
            p,
            np.pi / 4,
            (-np.pi / 9, np.pi / 4.5),
            (np.pi / 9, np.pi / 6),
            False,
            "nearest",
            "border",
            True,
            (1, 89, 105, 104),
        )
    )
    TEST_CASES_3D.append(
        (
            p,
            np.pi / 4,
            (-np.pi / 9, np.pi / 4.5),
            (np.pi / 9, np.pi / 6),
            False,
            GridSampleMode.NEAREST,
            GridSamplePadMode.BORDER,
            True,
            (1, 89, 105, 104),
        )
    )
    TEST_CASES_3D.append(
        (
            p,
            0.0,
            (2 * np.pi, 2.06 * np.pi),
            (-np.pi / 180, np.pi / 180),
            True,
            "nearest",
            "zeros",
            True,
            (1, 48, 64, 80),
        )
    )
    TEST_CASES_3D.append(
        (
            p,
            0.0,
            (2 * np.pi, 2.06 * np.pi),
            (-np.pi / 180, np.pi / 180),
            True,
            GridSampleMode.NEAREST,
            GridSamplePadMode.ZEROS,
            True,
            (1, 48, 64, 80),
        )
    )
    TEST_CASES_3D.append((p, (-np.pi / 4, 0), 0, 0, False, "nearest", "zeros", False, (1, 48, 77, 90)))
    TEST_CASES_3D.append(
        (p, (-np.pi / 4, 0), 0, 0, False, GridSampleMode.NEAREST, GridSamplePadMode.ZEROS, False, (1, 48, 77, 90))
    )


class TestRandRotated2D(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_correct_results(self, im_type, degrees, keep_size, mode, padding_mode, align_corners):
        rotate_fn = RandRotated(
            "img",
            range_x=degrees,
            prob=1.0,
            keep_size=keep_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=np.float64,
        )
        im = im_type(self.imt[0])
        rotate_fn.set_random_state(243)
        rotated = rotate_fn({"img": im, "seg": im_type(self.segn[0])})

        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        angle = rotate_fn.rand_rotate.x
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -np.rad2deg(angle), (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        test_local_inversion(rotate_fn, rotated, {"img": im}, "img")
        for k, v in rotated.items():
            rotated[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        expected = np.stack(expected).astype(np.float32)
        good = np.sum(np.isclose(expected, rotated["img"][0], atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 5, "diff at most 5 pixels")


class TestRandRotated3D(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_shapes(self, im_type, x, y, z, keep_size, mode, padding_mode, align_corners, expected):
        rotate_fn = RandRotated(
            "img",
            range_x=x,
            range_y=y,
            range_z=z,
            prob=1.0,
            keep_size=keep_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=np.float64,
        )
        rotate_fn.set_random_state(243)
        rotated = rotate_fn({"img": im_type(self.imt[0]), "seg": im_type(self.segn[0])})
        np.testing.assert_allclose(rotated["img"].shape, expected)


if __name__ == "__main__":
    unittest.main()
