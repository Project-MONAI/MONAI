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

import torch
from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms import CenterSpatialCropd, Compose, Flipd, SpatialPadd
from monai.transforms.lazy.functional import resample
from monai.utils import convert_to_tensor
from tests.test_utils import assert_allclose, get_arange_img


def rotate_90_2d():
    t = torch.eye(3)
    t[:, 0] = torch.FloatTensor([0, -1, 0])
    t[:, 1] = torch.FloatTensor([1, 0, 0])
    return t


RESAMPLE_FUNCTION_CASES = [
    (get_arange_img((3, 3)), rotate_90_2d(), [[0, 3, 6], [0, 3, 6], [0, 3, 6]]),
    (get_arange_img((3, 3)), torch.eye(3), get_arange_img((3, 3))[0]),
]


class TestResampleFunction(unittest.TestCase):
    @parameterized.expand(RESAMPLE_FUNCTION_CASES)
    def test_resample_function_impl(self, img, matrix, expected):
        out = resample(convert_to_tensor(img), matrix, {"lazy_shape": img.shape[1:], "lazy_padding_mode": "border"})
        assert_allclose(out[0], expected, type_test=False)

        img = convert_to_tensor(img, dtype=torch.uint8)
        out = resample(img, matrix, {"lazy_resample_mode": "auto", "lazy_dtype": torch.float})
        out_1 = resample(img, matrix, {"lazy_resample_mode": "other value", "lazy_dtype": torch.float})
        self.assertIs(out.dtype, out_1.dtype)  # testing dtype in different lazy_resample_mode

    @parameterized.expand(
        [
            ["flip", lambda: Flipd("img", spatial_axis=0)],
            ["crop", lambda: CenterSpatialCropd("img", (5, 4))],
            ["pad", lambda: SpatialPadd("img", (12, 12))],
        ]
    )
    def test_lazy_crop_pad_keeps_affine_rank(self, _, make_transform):
        """A 2D image with a 4x4 affine keeps its out-of-plane spacing and origin on the lazy crop/pad path."""
        affine = torch.diag(torch.tensor([1.5, 0.8, 2.0, 1.0], dtype=torch.float64))
        affine[:3, 3] = torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)
        img = MetaTensor(torch.rand(1, 10, 8), affine=affine)

        eager = Compose([make_transform()], lazy=False)({"img": img.clone()})["img"]
        lazy = Compose([make_transform()], lazy=True)({"img": img.clone()})["img"]

        self.assertEqual(lazy.affine.shape, (4, 4))
        assert_allclose(lazy.affine, eager.affine, type_test=False)
        assert_allclose(lazy, eager, type_test=False)


if __name__ == "__main__":
    unittest.main()
