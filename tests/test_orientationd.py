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
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Orientationd
from tests.utils import TEST_DEVICES

TESTS = []
for device in TEST_DEVICES:
    TESTS.append(
        [{"keys": "seg", "axcodes": "RAS"}, torch.ones((2, 1, 2, 3)), torch.eye(4), (2, 1, 2, 3), "RAS", *device]
    )
    # 3d
    TESTS.append(
        [
            {"keys": ["img", "seg"], "axcodes": "PLI"},
            torch.ones((2, 1, 2, 3)),
            torch.eye(4),
            (2, 2, 1, 3),
            "PLI",
            *device,
        ]
    )
    # 2d
    TESTS.append(
        [{"keys": ["img", "seg"], "axcodes": "PLI"}, torch.ones((2, 1, 3)), torch.eye(4), (2, 3, 1), "PLS", *device]
    )
    # 1d
    TESTS.append([{"keys": ["img", "seg"], "axcodes": "L"}, torch.ones((2, 3)), torch.eye(4), (2, 3), "LAS", *device])
    # canonical
    TESTS.append(
        [
            {"keys": ["img", "seg"], "as_closest_canonical": True},
            torch.ones((2, 1, 2, 3)),
            torch.eye(4),
            (2, 1, 2, 3),
            "RAS",
            *device,
        ]
    )

TESTS_TORCH = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TESTS_TORCH.append([{"keys": "seg", "axcodes": "RAS"}, torch.ones(2, 1, 2, 3), track_meta, *device])


class TestOrientationdCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_orntd(
        self, init_param, img: torch.Tensor, affine: Optional[torch.Tensor], expected_shape, expected_code, device
    ):
        ornt = Orientationd(**init_param)
        if affine is not None:
            img = MetaTensor(img, affine=affine)
        img = img.to(device)
        data = {k: img.clone() for k in ornt.keys}
        res = ornt(data)
        for k in ornt.keys:
            _im = res[k]
            self.assertIsInstance(_im, MetaTensor)
            np.testing.assert_allclose(_im.shape, expected_shape)
            code = nib.aff2axcodes(_im.affine.cpu(), ornt.ornt_transform.labels)
            self.assertEqual("".join(code), expected_code)

    @parameterized.expand(TESTS_TORCH)
    def test_orntd_torch(self, init_param, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        ornt = Orientationd(**init_param)
        img = img.to(device)
        expected_shape = img.shape
        expected_code = ornt.ornt_transform.axcodes
        data = {k: img.clone() for k in ornt.keys}
        res = ornt(data)
        for k in ornt.keys:
            _im = res[k]
            np.testing.assert_allclose(_im.shape, expected_shape)
            if track_meta:
                self.assertIsInstance(_im, MetaTensor)
                code = nib.aff2axcodes(_im.affine.cpu(), ornt.ornt_transform.labels)
                self.assertEqual("".join(code), expected_code)
            else:
                self.assertIsInstance(_im, torch.Tensor)
                self.assertNotIsInstance(_im, MetaTensor)


if __name__ == "__main__":
    unittest.main()
