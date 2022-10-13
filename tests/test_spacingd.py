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
from typing import List, Tuple

import numpy as np
import torch
from parameterized import parameterized

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.transforms import Spacingd
from tests.utils import TEST_DEVICES, assert_allclose

TESTS: List[Tuple] = []
for device in TEST_DEVICES:
    TESTS.append(
        (
            "spacing 3d",
            {"image": MetaTensor(torch.ones((2, 10, 15, 20)), affine=torch.eye(4))},
            dict(keys="image", pixdim=(1, 2, 1.4)),
            (2, 10, 8, 15),
            torch.as_tensor(np.diag([1, 2, 1.4, 1.0])),
            *device,
        )
    )
    TESTS.append(
        (
            "spacing 2d",
            {"image": MetaTensor(torch.ones((2, 10, 20)), affine=torch.eye(3))},
            dict(keys="image", pixdim=(1, 2)),
            (2, 10, 10),
            torch.as_tensor(np.diag((1, 2, 1))),
            *device,
        )
    )
    TESTS.append(
        (
            "spacing 2d no metadata",
            {"image": MetaTensor(torch.ones((2, 10, 20)))},
            dict(keys="image", pixdim=(1, 2)),
            (2, 10, 10),
            torch.as_tensor(np.diag((1, 2, 1))),
            *device,
        )
    )
    TESTS.append(
        (
            "interp all",
            {
                "image": MetaTensor(np.arange(20).reshape((2, 1, 10)), affine=torch.eye(4)),
                "seg": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
            },
            dict(keys=("image", "seg"), mode="nearest", pixdim=(1, 0.2)),
            (2, 1, 46),
            torch.as_tensor(np.diag((1, 0.2, 1))),
            *device,
        )
    )
    TESTS.append(
        (
            "interp sep",
            {
                "image": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
                "seg": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
            },
            dict(keys=("image", "seg"), mode=("bilinear", "nearest"), pixdim=(1, 0.2)),
            (2, 1, 46),
            torch.as_tensor(np.diag((1, 0.2, 1))),
            *device,
        )
    )

TESTS_TORCH = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TESTS_TORCH.append([{"keys": "seg", "pixdim": [0.2, 0.3, 1]}, torch.ones(2, 1, 2, 3), track_meta, *device])


class TestSpacingDCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_spacingd(self, _, data, kw_args, expected_shape, expected_affine, device):
        data = {k: v.to(device) for k, v in data.items()}
        res = Spacingd(**kw_args)(data)
        in_img = data["image"]
        out_img = res["image"]
        self.assertEqual(in_img.device, out_img.device)
        # no change in number of keys
        self.assertEqual(tuple(sorted(data)), tuple(sorted(res)))
        np.testing.assert_allclose(out_img.shape, expected_shape)
        assert_allclose(out_img.affine, expected_affine)

    @parameterized.expand(TESTS_TORCH)
    def test_orntd_torch(self, init_param, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        tr = Spacingd(**init_param)
        data = {"seg": img.to(device)}
        res = tr(data)["seg"]

        if track_meta:
            self.assertIsInstance(res, MetaTensor)
            new_spacing = affine_to_spacing(res.affine, 3)
            assert_allclose(new_spacing, init_param["pixdim"], type_test=False)
            self.assertNotEqual(img.shape, res.shape)
        else:
            self.assertIsInstance(res, torch.Tensor)
            self.assertNotIsInstance(res, MetaTensor)
            self.assertNotEqual(img.shape, res.shape)


if __name__ == "__main__":
    unittest.main()
