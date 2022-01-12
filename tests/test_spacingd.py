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
from monai.utils.enums import CommonKeys, DictPostFixes
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS: List[Tuple] = []
for p in TEST_NDARRAYS:
    TESTS.append(
        (
            "spacing 3d",
            {
                CommonKeys.IMAGE: p(np.ones((2, 10, 15, 20))),
                f"{CommonKeys.IMAGE}_{DictPostFixes.META}": {"affine": p(np.eye(4))},
            },
            dict(keys=CommonKeys.IMAGE, pixdim=(1, 2, 1.4)),
            (CommonKeys.IMAGE, f"{CommonKeys.IMAGE}_{DictPostFixes.META}", "image_transforms"),
            (2, 10, 8, 15),
            p(np.diag([1, 2, 1.4, 1.0])),
        )
    )
    TESTS.append(
        (
            "spacing 2d",
            {CommonKeys.IMAGE: np.ones((2, 10, 20)), f"{CommonKeys.IMAGE}_{DictPostFixes.META}": {"affine": np.eye(3)}},
            dict(keys=CommonKeys.IMAGE, pixdim=(1, 2)),
            (CommonKeys.IMAGE, f"{CommonKeys.IMAGE}_{DictPostFixes.META}", "image_transforms"),
            (2, 10, 10),
            np.diag((1, 2, 1)),
        )
    )
    TESTS.append(
        (
            "spacing 2d no metadata",
            {CommonKeys.IMAGE: np.ones((2, 10, 20))},
            dict(keys=CommonKeys.IMAGE, pixdim=(1, 2)),
            (CommonKeys.IMAGE, f"{CommonKeys.IMAGE}_{DictPostFixes.META}", "image_transforms"),
            (2, 10, 10),
            np.diag((1, 2, 1)),
        )
    )
    TESTS.append(
        (
            "interp all",
            {
                CommonKeys.IMAGE: np.arange(20).reshape((2, 1, 10)),
                "seg": np.ones((2, 1, 10)),
                f"{CommonKeys.IMAGE}_{DictPostFixes.META}": {"affine": np.eye(4)},
                "seg_meta_dict": {"affine": np.eye(4)},
            },
            dict(keys=(CommonKeys.IMAGE, "seg"), mode="nearest", pixdim=(1, 0.2)),
            (
                CommonKeys.IMAGE,
                f"{CommonKeys.IMAGE}_{DictPostFixes.META}",
                "image_transforms",
                "seg",
                "seg_meta_dict",
                "seg_transforms",
            ),
            (2, 1, 46),
            np.diag((1, 0.2, 1, 1)),
        )
    )
    TESTS.append(
        (
            "interp sep",
            {
                CommonKeys.IMAGE: np.ones((2, 1, 10)),
                "seg": np.ones((2, 1, 10)),
                f"{CommonKeys.IMAGE}_{DictPostFixes.META}": {"affine": np.eye(4)},
                "seg_meta_dict": {"affine": np.eye(4)},
            },
            dict(keys=(CommonKeys.IMAGE, "seg"), mode=("bilinear", "nearest"), pixdim=(1, 0.2)),
            (
                CommonKeys.IMAGE,
                f"{CommonKeys.IMAGE}_{DictPostFixes.META}",
                "image_transforms",
                "seg",
                "seg_meta_dict",
                "seg_transforms",
            ),
            (2, 1, 46),
            np.diag((1, 0.2, 1, 1)),
        )
    )


class TestSpacingDCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_spacingd(self, _, data, kw_args, expected_keys, expected_shape, expected_affine):
        res = Spacingd(**kw_args)(data)
        if isinstance(data[CommonKeys.IMAGE], torch.Tensor):
            self.assertEqual(data[CommonKeys.IMAGE].device, res[CommonKeys.IMAGE].device)
        self.assertEqual(expected_keys, tuple(sorted(res)))
        np.testing.assert_allclose(res[CommonKeys.IMAGE].shape, expected_shape)
        assert_allclose(res[f"{CommonKeys.IMAGE}_{DictPostFixes.META}"]["affine"], expected_affine)


if __name__ == "__main__":
    unittest.main()
