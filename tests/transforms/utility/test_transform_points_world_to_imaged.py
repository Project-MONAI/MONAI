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
from monai.transforms.utility.dictionary import ApplyTransformToPointsd, TransformPointsWorldToImaged

DATA_2D = torch.rand(1, 64, 64)
DATA_3D = torch.rand(1, 64, 64, 64)
POINT_2D_WORLD = torch.tensor([[[2, 2], [2, 4], [4, 6]]])
POINT_2D_IMAGE = torch.tensor([[[1, 1], [1, 2], [2, 3]]])
POINT_3D_WORLD = torch.tensor([[[2, 4, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])
POINT_3D_IMAGE = torch.tensor([[[-8, 8, 6], [-2, 14, 12]], [[4, 20, 18], [10, 26, 24]]])
POINT_2D_IMAGE_RAS = torch.tensor([[[-1, -1], [-1, -2], [-2, -3]]])
POINT_3D_IMAGE_RAS = torch.tensor([[[-12, 0, 6], [-18, -6, 12]], [[-24, -12, 18], [-30, -18, 24]]])
AFFINE_1 = torch.tensor([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
AFFINE_2 = torch.tensor([[1, 0, 0, 10], [0, 1, 0, -4], [0, 0, 1, 0], [0, 0, 0, 1]])

TEST_CASES = [
    # 2D: world -> image using image affine
    [MetaTensor(DATA_2D, affine=AFFINE_1), POINT_2D_WORLD, False, POINT_2D_IMAGE],
    # 3D: world -> image using image affine
    [MetaTensor(DATA_3D, affine=AFFINE_2), POINT_3D_WORLD, False, POINT_3D_IMAGE],
    # 2D with affine_lps_to_ras
    [MetaTensor(DATA_2D, affine=AFFINE_1), POINT_2D_WORLD, True, POINT_2D_IMAGE_RAS],
    # 3D with affine_lps_to_ras
    [MetaTensor(DATA_3D, affine=AFFINE_2), POINT_3D_WORLD, True, POINT_3D_IMAGE_RAS],
]


class TestTransformPointsWorldToImaged(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transform(self, image, points, affine_lps_to_ras, expected_output):
        data = {"image": image, "point": points}
        transform = TransformPointsWorldToImaged(
            keys="point", refer_keys="image", dtype=torch.int64, affine_lps_to_ras=affine_lps_to_ras
        )
        output = transform(data)
        self.assertTrue(torch.allclose(output["point"], expected_output))

    @parameterized.expand(TEST_CASES)
    def test_matches_base_class(self, image, points, affine_lps_to_ras, expected_output):
        """Verify that TransformPointsWorldToImaged produces the same result as
        ApplyTransformToPointsd with invert_affine=True."""
        data = {"image": image, "point": points}
        convenience = TransformPointsWorldToImaged(
            keys="point", refer_keys="image", dtype=torch.int64, affine_lps_to_ras=affine_lps_to_ras
        )
        base = ApplyTransformToPointsd(
            keys="point", refer_keys="image", dtype=torch.int64, invert_affine=True, affine_lps_to_ras=affine_lps_to_ras
        )
        out_convenience = convenience(dict(data))
        out_base = base(dict(data))
        self.assertTrue(torch.allclose(out_convenience["point"], out_base["point"]))

    @parameterized.expand(TEST_CASES)
    def test_inverse(self, image, points, affine_lps_to_ras, _expected_output):
        data = {"image": image, "point": points}
        transform = TransformPointsWorldToImaged(
            keys="point", refer_keys="image", dtype=torch.int64, affine_lps_to_ras=affine_lps_to_ras
        )
        output = transform(data)
        inverted = transform.inverse(output)
        self.assertTrue(torch.allclose(inverted["point"], points))

    def test_missing_refer_key(self):
        data = {"point": POINT_2D_WORLD}
        transform = TransformPointsWorldToImaged(keys="point", refer_keys="image", dtype=torch.int64)
        with self.assertRaises(KeyError):
            transform(data)


if __name__ == "__main__":
    unittest.main()
