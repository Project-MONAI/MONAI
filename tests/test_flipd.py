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

import numpy as np
import torch
from parameterized import parameterized

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Flipd
from tests.utils import TEST_DEVICES, TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

INVALID_CASES = [("wrong_axis", ["s", 1], TypeError), ("not_numbers", "s", TypeError)]

VALID_CASES = [("no_axis", None), ("one_axis", 1), ("many_axis", [0, 1])]

TORCH_CASES = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TORCH_CASES.append([[0, 1], torch.zeros((1, 3, 2)), track_meta, *device])


class TestFlipd(NumpyImageTestCase2D):
    @parameterized.expand(INVALID_CASES)
    def test_invalid_cases(self, _, spatial_axis, raises):
        with self.assertRaises(raises):
            flip = Flipd(keys="img", spatial_axis=spatial_axis)
            flip({"img": self.imt[0]})

    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, spatial_axis):
        for p in TEST_NDARRAYS_ALL:
            flip = Flipd(keys="img", spatial_axis=spatial_axis)
            expected = [np.flip(channel, spatial_axis) for channel in self.imt[0]]
            expected = np.stack(expected)
            im = p(self.imt[0])
            result = flip({"img": im})["img"]
            assert_allclose(result, p(expected), type_test="tensor")
            test_local_inversion(flip, {"img": result}, {"img": im}, "img")

    @parameterized.expand(TORCH_CASES)
    def test_torch(self, init_param, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        img = img.to(device)
        xform = Flipd("image", init_param)
        res = xform({"image": img})
        self.assertEqual(img.shape, res["image"].shape)
        if track_meta:
            self.assertIsInstance(res["image"], MetaTensor)
        else:
            self.assertNotIsInstance(res["image"], MetaTensor)
            self.assertIsInstance(res["image"], torch.Tensor)
            with self.assertRaisesRegex(ValueError, "MetaTensor"):
                xform.inverse(res)


if __name__ == "__main__":
    unittest.main()
