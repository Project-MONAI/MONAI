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

import torch
from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    RandAffined,
    Resize,
    Resized,
    ResizeWithPadOrCropd,
    Rotated,
    Spacingd,
    SpatialPadd,
    Zoomd,
)
from monai.transforms.lazy.utils import kwargs_from_pending
from monai.utils import LazyAttr, TraceKeys
from tests.test_utils import assert_allclose

TEST_CASES = [
    ["spacing", [Spacingd("label", pixdim=1.3, mode="nearest")]],
    ["spacing_pad", [Spacingd("label", pixdim=1.3, mode="nearest"), SpatialPadd("label", 40)]],
    ["pad_spacing", [SpatialPadd("label", 40), Spacingd("label", pixdim=1.3, mode="nearest")]],
    ["spacing_crop", [Spacingd("label", pixdim=0.7, mode="nearest"), CenterSpatialCropd("label", 30)]],
    ["rotate_padcrop", [Rotated("label", 0.3, mode="nearest"), ResizeWithPadOrCropd("label", 36)]],
    [
        "zoom_affine",
        [Zoomd("label", 1.2, mode="nearest"), RandAffined("label", prob=1.0, rotate_range=0.3, mode="nearest")],
    ],
    ["resize", [Resized("label", (20, 20, 20), mode="nearest")]],
]


class TestLazyInterpMode(unittest.TestCase):
    """Tests that lazy resampling honours the interpolation/padding modes set on each transform."""

    @parameterized.expand(TEST_CASES)
    def test_binary_label_stays_binary(self, _, transforms):
        """Check a binary label resampled lazily with ``mode="nearest"`` contains only 0 and 1.

        Args:
            _: name of the test case.
            transforms: dictionary transforms (all using ``mode="nearest"`` where applicable) to compose lazily.
        """
        torch.manual_seed(0)
        label = MetaTensor((torch.rand(1, 32, 32, 32) > 0.5).float(), affine=torch.eye(4))
        pipeline = Compose(transforms, lazy=True)
        pipeline.set_random_state(0)
        out = pipeline({"label": label})["label"]
        self.assertTrue(set(torch.unique(out).tolist()) <= {0.0, 1.0})

    def test_bilinear_key_still_interpolates(self):
        """Check per-key modes in one lazy pipeline: the bilinear key is interpolated, the nearest key stays binary."""
        torch.manual_seed(0)
        data = {
            "img": MetaTensor((torch.rand(1, 32, 32, 32) > 0.5).float(), affine=torch.eye(4)),
            "label": MetaTensor((torch.rand(1, 32, 32, 32) > 0.5).float(), affine=torch.eye(4)),
        }
        keys = ["img", "label"]
        pipeline = Compose([Spacingd(keys, pixdim=1.3, mode=("bilinear", "nearest")), SpatialPadd(keys, 40)], lazy=True)
        out = pipeline(data)
        self.assertTrue(set(torch.unique(out["label"]).tolist()) <= {0.0, 1.0})
        # a binary input only yields values strictly between 0 and 1 if it was actually interpolated
        self.assertTrue(((out["img"] > 0) & (out["img"] < 1)).any())

    def test_nearest_exact_resize_matches_eager(self):
        """Check lazy ``Resize(mode="nearest-exact")`` matches eager output when no sample lies on an exact tie."""
        torch.manual_seed(0)
        img = MetaTensor(torch.randint(0, 5, (1, 17, 23, 29)).float(), affine=torch.eye(4))
        eager = Resize((20, 20, 20), mode="nearest-exact")(img.clone())
        lazy = Compose([Resize((20, 20, 20), mode="nearest-exact")], lazy=True)(img.clone())
        assert_allclose(lazy, eager, type_test=False)

    def test_kwargs_from_pending_extra_info(self):
        """Check ``kwargs_from_pending`` reads, prioritises and normalises modes recorded in ``extra_info``."""
        kw = kwargs_from_pending({TraceKeys.EXTRA_INFO: {"mode": "nearest", "padding_mode": "border"}})
        self.assertEqual(kw[LazyAttr.INTERP_MODE], "nearest")
        self.assertEqual(kw[LazyAttr.PADDING_MODE], "border")
        # explicit lazy attributes take precedence
        kw = kwargs_from_pending({LazyAttr.INTERP_MODE: "bilinear", TraceKeys.EXTRA_INFO: {"mode": "nearest"}})
        self.assertEqual(kw[LazyAttr.INTERP_MODE], "bilinear")
        # padding modes recorded by pad transforms are not interpolation modes
        kw = kwargs_from_pending({TraceKeys.EXTRA_INFO: {"padded": ((0, 0),), "mode": "constant"}})
        self.assertIsNone(kw[LazyAttr.INTERP_MODE])
        # torch interpolate-only modes are mapped/ignored
        self.assertEqual(
            kwargs_from_pending({TraceKeys.EXTRA_INFO: {"mode": "nearest-exact"}})[LazyAttr.INTERP_MODE], "nearest"
        )
        self.assertEqual(
            kwargs_from_pending({TraceKeys.EXTRA_INFO: {"mode": "trilinear"}})[LazyAttr.INTERP_MODE], "bilinear"
        )
        self.assertIsNone(kwargs_from_pending({TraceKeys.EXTRA_INFO: {"mode": "area"}})[LazyAttr.INTERP_MODE])
        self.assertEqual(kwargs_from_pending({TraceKeys.EXTRA_INFO: {"mode": 0}})[LazyAttr.INTERP_MODE], 0)


if __name__ == "__main__":
    unittest.main()
