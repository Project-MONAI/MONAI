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

from monai.transforms import Affine, Rotate
from monai.transforms.utils import create_rotate
from monai.utils import optional_import

Rotation, has_scipy = optional_import("scipy.spatial.transform", name="Rotation")

RADIANS = (0.3, -0.7, 1.1)
SEQUENCES = ["xyz", "zyx", "zxy", "yxz", "yzx", "xzy", "XYZ", "ZYX", "ZXY", "xzx", "ZXZ"]
BAD_ORDERS = ["abc", "Xy", "xx", "wxyz", "", "xyzz"]


def _legacy_rotate_3d(radians):
    affine = np.eye(4)
    a = np.eye(4)
    a[1, 1], a[1, 2], a[2, 1], a[2, 2] = np.cos(radians[0]), -np.sin(radians[0]), np.sin(radians[0]), np.cos(radians[0])
    affine = affine @ a
    a = np.eye(4)
    a[0, 0], a[0, 2], a[2, 0], a[2, 2] = np.cos(radians[1]), np.sin(radians[1]), -np.sin(radians[1]), np.cos(radians[1])
    affine = affine @ a
    a = np.eye(4)
    a[0, 0], a[0, 1], a[1, 0], a[1, 1] = np.cos(radians[2]), -np.sin(radians[2]), np.sin(radians[2]), np.cos(radians[2])
    return affine @ a


class TestCreateRotateOrder(unittest.TestCase):
    def test_default_matches_legacy(self):
        legacy = _legacy_rotate_3d(RADIANS)
        np.testing.assert_allclose(np.asarray(create_rotate(3, RADIANS)), legacy, atol=1e-6)
        np.testing.assert_allclose(np.asarray(create_rotate(3, RADIANS, rotate_order="XYZ")), legacy, atol=1e-6)

    @parameterized.expand([(s,) for s in SEQUENCES])
    @unittest.skipUnless(has_scipy, "requires scipy")
    def test_matches_scipy(self, order):
        radians = RADIANS[: len(order)]
        expected = Rotation.from_euler(order, radians).as_matrix()
        np_mat = np.asarray(create_rotate(3, radians, rotate_order=order))[:3, :3]
        torch_mat = create_rotate(3, radians, rotate_order=order, backend="torch").cpu().numpy()[:3, :3]
        np.testing.assert_allclose(np_mat, expected, atol=1e-6)
        np.testing.assert_allclose(torch_mat, expected, atol=1e-5)

    @parameterized.expand([(b,) for b in BAD_ORDERS])
    def test_invalid_order_raises(self, order):
        with self.assertRaises(ValueError):
            create_rotate(3, RADIANS, rotate_order=order)

    def test_order_too_short_for_radians(self):
        with self.assertRaises(ValueError):
            create_rotate(3, RADIANS, rotate_order="xy")

    def test_2d_ignores_order(self):
        np.testing.assert_allclose(
            np.asarray(create_rotate(2, [0.5])), np.asarray(create_rotate(2, [0.5], rotate_order="x")), atol=1e-6
        )

    def test_transform_order_changes_output(self):
        img = torch.arange(8 * 9 * 10, dtype=torch.float32).reshape(1, 8, 9, 10)
        default = Rotate(angle=RADIANS, rotate_order="XYZ")(img)
        reordered = Rotate(angle=RADIANS, rotate_order="zyx")(img)
        self.assertFalse(torch.allclose(default, reordered))

    def test_transform_invertible_with_order(self):
        img = torch.arange(10 * 10 * 10, dtype=torch.float32).reshape(1, 10, 10, 10)
        rotate = Rotate(angle=RADIANS, rotate_order="zyx", keep_size=True)
        out = rotate(img)
        inv = rotate.inverse(out)
        self.assertEqual(tuple(inv.shape), tuple(img.shape))
        # rotation is lossy, so check the inverse undoes most of it rather than an exact round-trip
        err_inv = np.abs(np.asarray(inv.cpu()) - np.asarray(img)).mean()
        err_rot = np.abs(np.asarray(out.cpu()) - np.asarray(img)).mean()
        self.assertLess(err_inv, err_rot)

    def test_affine_propagates_order(self):
        img = torch.arange(6 * 7 * 8, dtype=torch.float32).reshape(1, 6, 7, 8)
        affine_default = Affine(rotate_params=RADIANS, image_only=True)(img)
        affine_reordered = Affine(rotate_params=RADIANS, rotate_order="zyx", image_only=True)(img)
        self.assertFalse(torch.allclose(affine_default, affine_reordered))


if __name__ == "__main__":
    unittest.main()
