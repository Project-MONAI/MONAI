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
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import CLAHE, CLAHEd
from monai.utils import get_equivalent_dtype
from tests.test_utils import TEST_NDARRAYS, SkipIfNoModule, assert_allclose


def _img_2d(seed=0):
    return np.random.default_rng(seed).random((1, 24, 24)).astype(np.float32)


def _brain(size=16, seed=0):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    c = (size - 1) / 2
    tissue = (((xx - c) / (size * 0.34)) ** 2 + ((yy - c) / (size * 0.44)) ** 2 + ((zz - c) / (size * 0.38)) ** 2) <= 1
    vol = np.zeros((1, size, size, size), np.float32)
    vol[0][tissue] = 300.0 + rng.normal(0, 40, tissue.sum())
    return vol, tissue


@SkipIfNoModule("SimpleITK")
class TestCLAHEd(unittest.TestCase):
    @parameterized.expand(
        [
            ({"keys": "img"},),
            ({"keys": "img", "radius": 3},),
            ({"keys": "img", "alpha": 0.0, "beta": 0.0},),
            ({"keys": "img", "nonzero": True},),
        ]
    )
    def test_matches_array_transform(self, arguments):
        img = _img_2d()
        result = CLAHEd(**arguments)({"img": img})["img"]
        array_arguments = {k: v for k, v in arguments.items() if k != "keys"}
        assert_allclose(result, CLAHE(**array_arguments)(img), type_test=False, rtol=1e-5, atol=1e-5)

    @parameterized.expand(TEST_NDARRAYS)
    def test_container_agnostic(self, im_type):
        img = _img_2d()
        result = CLAHEd(keys="img")({"img": im_type(img)})["img"]
        self.assertIsInstance(result, MetaTensor)
        assert_allclose(result, CLAHE()(img), type_test=False, rtol=1e-5, atol=1e-5)

    def test_multiple_keys(self):
        data = {"img": _img_2d(), "img2": _img_2d(seed=1)}
        result = CLAHEd(keys=["img", "img2"])(data)
        for key in ("img", "img2"):
            assert_allclose(result[key], CLAHE()(data[key]), type_test=False, rtol=1e-5, atol=1e-5)

    def test_other_keys_untouched(self):
        result = CLAHEd(keys="img")({"img": _img_2d(), "label": 7})
        self.assertEqual(result["label"], 7)

    def test_allow_missing_keys(self):
        result = CLAHEd(keys=["img", "missing"], allow_missing_keys=True)({"img": _img_2d()})
        self.assertIn("img", result)
        self.assertNotIn("missing", result)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            CLAHEd(keys=["img", "missing"])({"img": _img_2d()})

    def test_mask_key(self):
        vol, tissue = _brain()
        result = CLAHEd(keys="img", mask_key="brain")({"img": vol, "brain": tissue})["img"]
        assert_allclose(result, CLAHE(mask=tissue)(vol), type_test=False, rtol=1e-5, atol=1e-5)

    def test_mask_key_hands_back_background(self):
        vol, tissue = _brain()
        result = CLAHEd(keys="img", mask_key="brain")({"img": vol, "brain": tissue})["img"]
        assert_allclose(result.numpy()[0][~tissue], vol[0][~tissue], type_test=False)

    def test_constructor_mask_disables_mask_key(self):
        _, tissue = _brain()
        self.assertIsNone(CLAHEd(keys="img", mask=tissue, mask_key="brain").mask_key)

    def test_mask_shared_across_keys(self):
        vol, tissue = _brain()
        data = {"t1": vol, "flair": vol * 3.0, "brain": tissue}
        result = CLAHEd(keys=["t1", "flair"], mask_key="brain")(data)
        for key in ("t1", "flair"):
            assert_allclose(result[key].numpy()[0][~tissue], data[key][0][~tissue], type_test=False)

    @parameterized.expand([((1, 10, 12, 8),), ((4, 8, 8, 8),)])
    def test_3d_shape_preserved(self, shape):
        vol = np.random.default_rng(0).random(shape).astype(np.float32)
        self.assertTupleEqual(tuple(CLAHEd(keys="img", radius=2)({"img": vol})["img"].shape), shape)

    @parameterized.expand([(np.float32,), (np.float64,), (None,)])
    def test_dtype(self, dtype):
        result = CLAHEd(keys="img", dtype=dtype)({"img": _img_2d()})["img"]
        self.assertEqual(
            get_equivalent_dtype(result.dtype, data_type=np.ndarray), dtype if dtype is not None else np.float32
        )

    def test_invalid_alpha_fails_at_construction(self):
        with self.assertRaises(ValueError):
            CLAHEd(keys="img", alpha=1.5)

    def test_constant_image(self):
        img = np.full((1, 12, 12), 0.5, dtype=np.float32)
        out = CLAHEd(keys="img")({"img": img})["img"]
        assert_allclose(out[0], img[0], type_test=False)

    def test_num_threads_does_not_change_the_result(self):
        img = _img_2d()
        a = CLAHEd(keys="img", num_threads=1)({"img": img})["img"]
        b = CLAHEd(keys="img")({"img": img})["img"]
        assert_allclose(a, b, type_test=False, rtol=1e-5, atol=1e-5)

    def test_non_finite_input(self):
        img = _img_2d()
        img[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            CLAHEd(keys="img")({"img": img})


if __name__ == "__main__":
    unittest.main()
