# Copyright 2020 MONAI Consortium
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

from monai.transforms import Spacingd


class TestSpacingDCase(unittest.TestCase):
    def test_spacingd_3d(self):
        data = {"image": np.ones((2, 10, 15, 20)), "image.affine": np.eye(4)}
        spacing = Spacingd(keys="image", pixdim=(1, 2, 1.4))
        res = spacing(data)
        self.assertEqual(("image", "image.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 10, 8, 15))
        np.testing.assert_allclose(res["image.affine"], np.diag([1, 2, 1.4, 1.0]))

    def test_spacingd_2d(self):
        data = {"image": np.ones((2, 10, 20)), "image.affine": np.eye(3)}
        spacing = Spacingd(keys="image", pixdim=(1, 2, 1.4))
        res = spacing(data)
        self.assertEqual(("image", "image.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 10, 10))
        np.testing.assert_allclose(res["image.affine"], np.diag((1, 2, 1)))

    def test_spacingd_1d(self):
        data = {"image": np.arange(20).reshape((2, 10)), "image.original_affine": np.diag((3, 2, 1, 1))}
        data["image.affine"] = data["image.original_affine"]
        spacing = Spacingd(keys="image", pixdim=(0.2,))
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "image.original_affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 136))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 2, 1, 1)))
        np.testing.assert_allclose(res["image.original_affine"], np.diag((3, 2, 1, 1)))

    def test_interp_all(self):
        data = {
            "image": np.arange(20).reshape((2, 10)),
            "seg": np.ones((2, 10)),
            "image.affine": np.eye(4),
            "seg.affine": np.eye(4),
        }
        spacing = Spacingd(keys=("image", "seg"), interp_order=0, pixdim=(0.2,))
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))

    def test_interp_sep(self):
        data = {"image": np.ones((2, 10)), "seg": np.ones((2, 10)), "image.affine": np.eye(4), "seg.affine": np.eye(4)}
        spacing = Spacingd(keys=("image", "seg"), interp_order=(2, 0), pixdim=(0.2,))
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))


if __name__ == "__main__":
    unittest.main()
