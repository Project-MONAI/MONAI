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

import nibabel as nib
import numpy as np

from monai.transforms import Orientationd
from monai.utils.enums import PostFix
from tests.utils import TEST_NDARRAYS


class TestOrientationdCase(unittest.TestCase):
    def test_orntd(self):
        data = {"seg": np.ones((2, 1, 2, 3)), PostFix.meta("seg"): {"affine": np.eye(4)}}
        ornt = Orientationd(keys="seg", axcodes="RAS")
        res = ornt(data)
        np.testing.assert_allclose(res["seg"].shape, (2, 1, 2, 3))
        code = nib.aff2axcodes(res[PostFix.meta("seg")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))

    def test_orntd_3d(self):
        for p in TEST_NDARRAYS:
            data = {
                "seg": p(np.ones((2, 1, 2, 3))),
                "img": p(np.ones((2, 1, 2, 3))),
                PostFix.meta("seg"): {"affine": np.eye(4)},
                PostFix.meta("img"): {"affine": np.eye(4)},
            }
            ornt = Orientationd(keys=("img", "seg"), axcodes="PLI")
            res = ornt(data)
            np.testing.assert_allclose(res["img"].shape, (2, 2, 1, 3))
            np.testing.assert_allclose(res["seg"].shape, (2, 2, 1, 3))
            code = nib.aff2axcodes(res[PostFix.meta("seg")]["affine"], ornt.ornt_transform.labels)
            self.assertEqual(code, ("P", "L", "I"))
            code = nib.aff2axcodes(res[PostFix.meta("img")]["affine"], ornt.ornt_transform.labels)
            self.assertEqual(code, ("P", "L", "I"))

    def test_orntd_2d(self):
        data = {
            "seg": np.ones((2, 1, 3)),
            "img": np.ones((2, 1, 3)),
            PostFix.meta("seg"): {"affine": np.eye(4)},
            PostFix.meta("img"): {"affine": np.eye(4)},
        }
        ornt = Orientationd(keys=("img", "seg"), axcodes="PLI")
        res = ornt(data)
        np.testing.assert_allclose(res["img"].shape, (2, 3, 1))
        code = nib.aff2axcodes(res[PostFix.meta("seg")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("P", "L", "S"))
        code = nib.aff2axcodes(res[PostFix.meta("img")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("P", "L", "S"))

    def test_orntd_1d(self):
        data = {
            "seg": np.ones((2, 3)),
            "img": np.ones((2, 3)),
            PostFix.meta("seg"): {"affine": np.eye(4)},
            PostFix.meta("img"): {"affine": np.eye(4)},
        }
        ornt = Orientationd(keys=("img", "seg"), axcodes="L")
        res = ornt(data)
        np.testing.assert_allclose(res["img"].shape, (2, 3))
        code = nib.aff2axcodes(res[PostFix.meta("seg")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("L", "A", "S"))
        code = nib.aff2axcodes(res[PostFix.meta("img")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("L", "A", "S"))

    def test_orntd_canonical(self):
        data = {
            "seg": np.ones((2, 1, 2, 3)),
            "img": np.ones((2, 1, 2, 3)),
            PostFix.meta("seg"): {"affine": np.eye(4)},
            PostFix.meta("img"): {"affine": np.eye(4)},
        }
        ornt = Orientationd(keys=("img", "seg"), as_closest_canonical=True)
        res = ornt(data)
        np.testing.assert_allclose(res["img"].shape, (2, 1, 2, 3))
        np.testing.assert_allclose(res["seg"].shape, (2, 1, 2, 3))
        code = nib.aff2axcodes(res[PostFix.meta("seg")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))
        code = nib.aff2axcodes(res[PostFix.meta("img")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))

    def test_orntd_no_metadata(self):
        data = {"seg": np.ones((2, 1, 2, 3))}
        ornt = Orientationd(keys="seg", axcodes="RAS")
        res = ornt(data)
        np.testing.assert_allclose(res["seg"].shape, (2, 1, 2, 3))
        code = nib.aff2axcodes(res[PostFix.meta("seg")]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))


if __name__ == "__main__":
    unittest.main()
