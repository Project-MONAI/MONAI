# Copyright 2020 - 2021 MONAI Consortium
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
from parameterized import parameterized

from monai.transforms import Orientationd
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:
        TESTS.append((p, q))


class TestOrientationdCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_orntd(self, im_type, affine_type):
        data = {"seg": im_type(np.ones((2, 1, 2, 3))), "seg_meta_dict": {"affine": affine_type(np.eye(4))}}
        ornt = Orientationd(keys="seg", axcodes="RAS")
        res = ornt(data)
        np.testing.assert_allclose(res["seg"].shape, (2, 1, 2, 3))
        code = nib.aff2axcodes(res["seg_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))

    @parameterized.expand(TESTS)
    def test_orntd_3d(self, im_type, affine_type):
        data = {
            "seg": im_type(np.ones((2, 1, 2, 3))),
            "img": im_type(np.ones((2, 1, 2, 3))),
            "seg_meta_dict": {"affine": affine_type(np.eye(4))},
            "img_meta_dict": {"affine": affine_type(np.eye(4))},
        }
        ornt = Orientationd(keys=("img", "seg"), axcodes="PLI")
        res = ornt(data)
        np.testing.assert_allclose(res["img"].shape, (2, 2, 1, 3))
        np.testing.assert_allclose(res["seg"].shape, (2, 2, 1, 3))
        code = nib.aff2axcodes(res["seg_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("P", "L", "I"))
        code = nib.aff2axcodes(res["img_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("P", "L", "I"))

    @parameterized.expand(TESTS)
    def test_orntd_2d(self, im_type, affine_type):
        data = {
            "seg": im_type(np.ones((2, 1, 3))),
            "img": im_type(np.ones((2, 1, 3))),
            "seg_meta_dict": {"affine": affine_type(np.eye(4))},
            "img_meta_dict": {"affine": affine_type(np.eye(4))},
        }
        ornt = Orientationd(keys=("img", "seg"), axcodes="PLI")
        res = ornt(data)
        np.testing.assert_allclose(res["img"].shape, (2, 3, 1))
        code = nib.aff2axcodes(res["seg_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("P", "L", "S"))
        code = nib.aff2axcodes(res["img_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("P", "L", "S"))

    @parameterized.expand(TESTS)
    def test_orntd_1d(self, im_type, affine_type):
        data = {
            "seg": im_type(np.ones((2, 3))),
            "img": im_type(np.ones((2, 3))),
            "seg_meta_dict": {"affine": affine_type(np.eye(4))},
            "img_meta_dict": {"affine": affine_type(np.eye(4))},
        }
        ornt = Orientationd(keys=("img", "seg"), axcodes="L")
        res = ornt(data)
        np.testing.assert_allclose(res["img"].shape, (2, 3))
        code = nib.aff2axcodes(res["seg_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("L", "A", "S"))
        code = nib.aff2axcodes(res["img_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("L", "A", "S"))

    @parameterized.expand(TESTS)
    def test_orntd_canonical(self, im_type, affine_type):
        data = {
            "seg": im_type(np.ones((2, 1, 2, 3))),
            "img": im_type(np.ones((2, 1, 2, 3))),
            "seg_meta_dict": {"affine": affine_type(np.eye(4))},
            "img_meta_dict": {"affine": affine_type(np.eye(4))},
        }
        ornt = Orientationd(keys=("img", "seg"), as_closest_canonical=True)
        res = ornt(data)
        np.testing.assert_allclose(res["img"].shape, (2, 1, 2, 3))
        np.testing.assert_allclose(res["seg"].shape, (2, 1, 2, 3))
        code = nib.aff2axcodes(res["seg_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))
        code = nib.aff2axcodes(res["img_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))

    @parameterized.expand(TESTS)
    def test_orntd_no_metadata(self, im_type, _):
        data = {"seg": im_type(np.ones((2, 1, 2, 3)))}
        ornt = Orientationd(keys="seg", axcodes="RAS")
        res = ornt(data)
        np.testing.assert_allclose(res["seg"].shape, (2, 1, 2, 3))
        code = nib.aff2axcodes(res["seg_meta_dict"]["affine"], ornt.ornt_transform.labels)
        self.assertEqual(code, ("R", "A", "S"))


if __name__ == "__main__":
    unittest.main()
