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
from monai.utils import set_determinism
from monai.utils.enums import CoordinateTransformMode
from monai.transforms.utility.array import CoordinateTransform, CoordinateTransformMode

set_determinism(seed=0)

DATA_2D = torch.rand(1, 64, 64)
DATA_3D = torch.rand(1, 64, 64, 64)
POINT_2D_WORLD = torch.tensor([[[2, 2], [2, 4], [4, 6]]])
POINT_2D_IMAGE = torch.tensor([[[1, 1], [1, 2], [2, 3]]])
POINT_3D_WORLD = torch.tensor([[[2, 4, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])
POINT_3D_IMAGE = torch.tensor([[[-8,  7,  6], [-2, 13, 12]], [[4, 19, 18], [10, 25, 24]]])

TEST_CASES = [
    [
        MetaTensor(DATA_2D, affine=torch.tensor([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
        POINT_2D_WORLD,
        CoordinateTransformMode.WORLD_TO_IMAGE,
        POINT_2D_IMAGE,
    ],
    [
        None,
        MetaTensor(POINT_2D_IMAGE, affine=torch.tensor([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
        CoordinateTransformMode.IMAGE_TO_WORLD,
        POINT_2D_WORLD,
    ],
    [
        MetaTensor(DATA_3D, affine=torch.tensor([[1, 0, 0, 10], [0, 1, 0, -3], [0, 0, 1, 0], [0, 0, 0, 1]])),
        POINT_3D_WORLD,
        CoordinateTransformMode.WORLD_TO_IMAGE,
        torch.tensor([[[-8,  7,  6], [-2, 13, 12]], [[4, 19, 18], [10, 25, 24]]]),
    ],
    [
        MetaTensor(DATA_3D, affine=torch.tensor([[1, 0, 0, 10], [0, 1, 0, -3], [0, 0, 1, 0], [0, 0, 0, 1]])),
        MetaTensor(POINT_3D_IMAGE, affine=torch.tensor([[1, 0, 0, 10], [0, 1, 0, -3], [0, 0, 1, 0], [0, 0, 0, 1]])),
        CoordinateTransformMode.IMAGE_TO_WORLD,
        POINT_3D_WORLD,
    ],
]



class TestCoordinateTransform(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transform_coordinates(self, image, points, mode, expected_output):
        transform = CoordinateTransform(dtype=torch.int64, mode=mode)
        affine = image.affine if image is not None else None
        output = transform(points, affine)
        self.assertTrue(torch.allclose(output, expected_output))

        invert_out = transform.inverse(output)
        print(invert_out, points, output.affine)
        self.assertTrue(torch.allclose(invert_out, points))


if __name__ == "__main__":
    unittest.main()
