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

from monai.transforms import Spacingd
from monai.utils.enums import PostFix
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS: List[Tuple] = []
for p in TEST_NDARRAYS:
    TESTS.append(
        (
            "spacing 3d",
            {"image": p(np.ones((2, 10, 15, 20))), PostFix.meta("image"): {"affine": p(np.eye(4))}},
            dict(keys="image", pixdim=(1, 2, 1.4)),
            ("image", PostFix.meta("image"), "image_transforms"),
            (2, 10, 8, 15),
            p(np.diag([1, 2, 1.4, 1.0])),
        )
    )
    TESTS.append(
        (
            "spacing 2d",
            {"image": np.ones((2, 10, 20)), PostFix.meta("image"): {"affine": np.eye(3)}},
            dict(keys="image", pixdim=(1, 2)),
            ("image", PostFix.meta("image"), "image_transforms"),
            (2, 10, 10),
            np.diag((1, 2, 1)),
        )
    )
    TESTS.append(
        (
            "spacing 2d no metadata",
            {"image": np.ones((2, 10, 20))},
            dict(keys="image", pixdim=(1, 2)),
            ("image", PostFix.meta("image"), "image_transforms"),
            (2, 10, 10),
            np.diag((1, 2, 1)),
        )
    )
    TESTS.append(
        (
            "interp all",
            {
                "image": np.arange(20).reshape((2, 1, 10)),
                "seg": np.ones((2, 1, 10)),
                PostFix.meta("image"): {"affine": np.eye(4)},
                PostFix.meta("seg"): {"affine": np.eye(4)},
            },
            dict(keys=("image", "seg"), mode="nearest", pixdim=(1, 0.2)),
            ("image", PostFix.meta("image"), "image_transforms", "seg", PostFix.meta("seg"), "seg_transforms"),
            (2, 1, 46),
            np.diag((1, 0.2, 1, 1)),
        )
    )
    TESTS.append(
        (
            "interp sep",
            {
                "image": np.ones((2, 1, 10)),
                "seg": np.ones((2, 1, 10)),
                PostFix.meta("image"): {"affine": np.eye(4)},
                PostFix.meta("seg"): {"affine": np.eye(4)},
            },
            dict(keys=("image", "seg"), mode=("bilinear", "nearest"), pixdim=(1, 0.2)),
            ("image", PostFix.meta("image"), "image_transforms", "seg", PostFix.meta("seg"), "seg_transforms"),
            (2, 1, 46),
            np.diag((1, 0.2, 1, 1)),
        )
    )


class TestSpacingDCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_spacingd(self, _, data, kw_args, expected_keys, expected_shape, expected_affine):
        res = Spacingd(**kw_args)(data)
        if isinstance(data["image"], torch.Tensor):
            self.assertEqual(data["image"].device, res["image"].device)
        self.assertEqual(expected_keys, tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, expected_shape)
        assert_allclose(res[PostFix.meta("image")]["affine"], expected_affine)


if __name__ == "__main__":
    unittest.main()
