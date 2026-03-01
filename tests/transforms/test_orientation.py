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
from typing import cast

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Orientation, create_rotate, create_translate
from monai.utils import SpaceKeys
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.test_utils import TEST_DEVICES, assert_allclose

TESTS = []
for device in TEST_DEVICES:
    TESTS.append(
        [
            {"axcodes": "RAS"},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.eye(4),
            torch.arange(12).reshape((2, 1, 2, 3)),
            "RAS",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "LPS"},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.eye(4),
            torch.arange(12).reshape((2, 1, 2, 3)),
            "LPS",
            True,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "ALS"},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.as_tensor(np.diag([-1, -1, 1, 1])),
            torch.tensor([[[[3, 4, 5]], [[0, 1, 2]]], [[[9, 10, 11]], [[6, 7, 8]]]]),
            "ALS",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "PRS"},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.as_tensor(np.diag([-1, -1, 1, 1])),
            torch.tensor([[[[3, 4, 5]], [[0, 1, 2]]], [[[9, 10, 11]], [[6, 7, 8]]]]),
            "PRS",
            True,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "RAS"},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.as_tensor(np.diag([-1, -1, 1, 1])),
            torch.tensor([[[[3, 4, 5], [0, 1, 2]]], [[[9, 10, 11], [6, 7, 8]]]]),
            "RAS",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "LPS"},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.as_tensor(np.diag([-1, -1, 1, 1])),
            torch.tensor([[[[3, 4, 5], [0, 1, 2]]], [[[9, 10, 11], [6, 7, 8]]]]),
            "LPS",
            True,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "AL"},
            torch.arange(6).reshape((2, 1, 3)),
            torch.eye(3),
            torch.tensor([[[0], [1], [2]], [[3], [4], [5]]]),
            "AL",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "PR"},
            torch.arange(6).reshape((2, 1, 3)),
            torch.eye(3),
            torch.tensor([[[0], [1], [2]], [[3], [4], [5]]]),
            "PR",
            True,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "L"},
            torch.arange(6).reshape((2, 3)),
            torch.eye(2),
            torch.tensor([[2, 1, 0], [5, 4, 3]]),
            "L",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "R"},
            torch.arange(6).reshape((2, 3)),
            torch.eye(2),
            torch.tensor([[2, 1, 0], [5, 4, 3]]),
            "R",
            True,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "L"},
            torch.arange(6).reshape((2, 3)),
            torch.eye(2),
            torch.tensor([[2, 1, 0], [5, 4, 3]]),
            "L",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "L"},
            torch.arange(6).reshape((2, 3)),
            torch.as_tensor(np.diag([-1, 1])),
            torch.arange(6).reshape((2, 3)),
            "L",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "LPS"},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.as_tensor(
                create_translate(3, (10, 20, 30))
                @ create_rotate(3, (np.pi / 2, np.pi / 2, np.pi / 4))
                @ np.diag([-1, 1, 1, 1])
            ),
            torch.tensor([[[[2, 5]], [[1, 4]], [[0, 3]]], [[[8, 11]], [[7, 10]], [[6, 9]]]]),
            "LPS",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"as_closest_canonical": True},
            torch.arange(12).reshape((2, 1, 2, 3)),
            torch.as_tensor(
                create_translate(3, (10, 20, 30))
                @ create_rotate(3, (np.pi / 2, np.pi / 2, np.pi / 4))
                @ np.diag([-1, 1, 1, 1])
            ),
            torch.tensor([[[[0, 3]], [[1, 4]], [[2, 5]]], [[[6, 9]], [[7, 10]], [[8, 11]]]]),
            "RAS",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"as_closest_canonical": True},
            torch.arange(6).reshape((1, 2, 3)),
            torch.as_tensor(create_translate(2, (10, 20)) @ create_rotate(2, (np.pi / 3)) @ np.diag([-1, -0.2, 1])),
            torch.tensor([[[3, 0], [4, 1], [5, 2]]]),
            "RA",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "LP"},
            torch.arange(6).reshape((1, 2, 3)),
            torch.as_tensor(create_translate(2, (10, 20)) @ create_rotate(2, (np.pi / 3)) @ np.diag([-1, -0.2, 1])),
            torch.tensor([[[2, 5], [1, 4], [0, 3]]]),
            "LP",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"axcodes": "LPID", "labels": tuple(zip("LPIC", "RASD"))},
            torch.zeros((1, 2, 3, 4, 5)),
            torch.as_tensor(np.diag([-1, -0.2, -1, 1, 1])),
            torch.zeros((1, 2, 3, 4, 5)),
            "LPID",
            False,
            *device,
        ]
    )
    TESTS.append(
        [
            {"as_closest_canonical": True, "labels": tuple(zip("LPIC", "RASD"))},
            torch.zeros((1, 2, 3, 4, 5)),
            torch.as_tensor(np.diag([-1, -0.2, -1, 1, 1])),
            torch.zeros((1, 2, 3, 4, 5)),
            "RASD",
            False,
            *device,
        ]
    )

TESTS_TORCH = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TESTS_TORCH.append([{"axcodes": "LPS"}, torch.zeros((1, 3, 4, 5)), track_meta, *device])

ILL_CASES = [
    # too short axcodes
    [{"axcodes": "RA"}, torch.arange(12).reshape((2, 1, 2, 3)), torch.eye(4)]
]

TESTS_INVERSE = []
for device in TEST_DEVICES:
    TESTS_INVERSE.append([True, *device])
    TESTS_INVERSE.append([False, *device])


class TestOrientationCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_ornt_meta(
        self,
        init_param,
        img: torch.Tensor,
        affine: torch.Tensor,
        expected_data: torch.Tensor,
        expected_code: str,
        lps_convention: bool,
        device,
    ):
        meta = {"space": SpaceKeys.LPS} if lps_convention else None
        img = MetaTensor(img, affine=affine, meta=meta).to(device)
        ornt = Orientation(**init_param)
        call_param = {"data_array": img}
        res = ornt(**call_param)  # type: ignore[arg-type]
        if img.ndim in (3, 4):
            test_resampler_lazy(ornt, res, init_param, call_param)

        assert_allclose(res, expected_data.to(device))
        labels = (("R", "L"), ("A", "P"), ("I", "S")) if lps_convention else ornt.labels
        new_code = nib.orientations.aff2axcodes(res.affine.cpu(), labels=labels)  # type: ignore
        self.assertEqual("".join(new_code), expected_code)

    @parameterized.expand(TESTS_TORCH)
    def test_ornt_torch(self, init_param, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        ornt = Orientation(**init_param)

        img = img.to(device)
        expected_data = img.clone()
        expected_code = ornt.axcodes

        res = ornt(img)
        assert_allclose(res, expected_data)
        if track_meta:
            self.assertIsInstance(res, MetaTensor)
            assert isinstance(res, MetaTensor)  # for mypy type narrowing
            new_code = nib.orientations.aff2axcodes(res.affine.cpu(), labels=ornt.labels)
            self.assertEqual("".join(new_code), expected_code)
        else:
            self.assertIsInstance(res, torch.Tensor)
            self.assertNotIsInstance(res, MetaTensor)

    @parameterized.expand(ILL_CASES)
    def test_bad_params(self, init_param, img: torch.Tensor, affine: torch.Tensor):
        img = MetaTensor(img, affine=affine)
        with self.assertRaises(ValueError):
            Orientation(**init_param)(img)

    @parameterized.expand(TESTS_INVERSE)
    def test_inverse(self, lps_convention: bool, device):
        img_t = torch.rand((1, 10, 9, 8), dtype=torch.float32, device=device)
        affine = torch.tensor(
            [[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cpu"
        )
        meta = {"fname": "somewhere", "space": SpaceKeys.LPS if lps_convention else SpaceKeys.RAS}
        img = MetaTensor(img_t, affine=affine, meta=meta)
        tr = Orientation("LPS")
        # check that image and affine have changed
        img = cast(MetaTensor, tr(img))
        self.assertNotEqual(img.shape, img_t.shape)
        self.assertGreater(float((affine - img.affine).max()), 0.5)
        # check that with inverse, image affine are back to how they were
        img = cast(MetaTensor, tr.inverse(img))
        self.assertEqual(img.shape, img_t.shape)
        self.assertLess(float((affine - img.affine).max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
