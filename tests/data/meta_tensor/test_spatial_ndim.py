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
from copy import deepcopy
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.data import MetaTensor, get_spatial_ndim
from monai.data.utils import collate_meta_tensor_fn, decollate_batch
from monai.transforms import Affine, LabelToContour, RandAffine, RandZoom, Resize, Rotate, SqueezeDim
from monai.transforms.utility.array import SplitDim
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

# (shape, affine, expected_spatial_ndim)
CONSTRUCTION_CASES = [
    ((1, 10, 10, 10), None, 3),  # default eye(4)
    ((1, 10, 10), torch.eye(3), 2),  # eye(3)
    ((1, 10), torch.eye(2), 1),  # eye(2)
]

# (description, op, expected_spatial_ndim)  -- op takes a 2D MetaTensor and returns a new one
PRESERVATION_CASES = [
    ("reshape", lambda t: t.reshape(1, 100), 2),
    ("unsqueeze", lambda t: t.unsqueeze(0), 2),
    ("squeeze", lambda t: t.unsqueeze(1).squeeze(1), 2),
    ("clone", lambda t: t.clone(), 2),
    ("deepcopy", lambda t: deepcopy(t), 2),
]


class TestSpatialNdim(unittest.TestCase):
    @parameterized.expand(CONSTRUCTION_CASES)
    def test_construction(self, shape, affine, expected):
        kwargs = {"affine": affine} if affine is not None else {}
        t = MetaTensor(torch.randn(*shape), **kwargs)
        self.assertEqual(t.spatial_ndim, expected)

    @parameterized.expand(PRESERVATION_CASES)
    def test_preserved_through_op(self, _desc, op, expected):
        t = MetaTensor(torch.randn(1, 10, 10), affine=torch.eye(3))
        t2 = op(t)
        self.assertEqual(t2.spatial_ndim, expected)

    def test_setter_and_validation(self):
        t = MetaTensor(torch.randn(1, 10, 10, 10))
        t.spatial_ndim = 2
        self.assertEqual(t.spatial_ndim, 2)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                t.spatial_ndim = bad

    def test_affine_setter_syncs(self):
        t = MetaTensor(torch.randn(1, 10, 10, 10))
        t.affine = torch.eye(3)
        self.assertEqual(t.spatial_ndim, 2)

    def test_copy_from_meta_tensor(self):
        t1 = MetaTensor(torch.randn(1, 10, 10), affine=torch.eye(3))
        self.assertEqual(MetaTensor(t1).spatial_ndim, 2)

    def test_collate_and_decollate(self):
        t1 = MetaTensor(torch.randn(1, 10, 10), affine=torch.eye(3))
        t2 = MetaTensor(torch.randn(1, 10, 10), affine=torch.eye(3))
        batch = collate_meta_tensor_fn([t1, t2])
        self.assertEqual(batch.spatial_ndim, 2)
        for item in decollate_batch(batch):
            self.assertIsInstance(item, MetaTensor)
            self.assertEqual(item.spatial_ndim, 2)

    def test_derived_properties(self):
        """peek_pending_rank, peek_pending_shape, and pixdim all respect spatial_ndim."""
        aff = torch.diag(torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64))
        t = MetaTensor(torch.randn(1, 10, 10), affine=aff)
        self.assertEqual(t.peek_pending_rank(), 2)
        self.assertEqual(t.peek_pending_shape(), (10, 10))
        self.assertEqual(len(t.pixdim), 2)

    def test_squeeze_dim_transform(self):
        t = MetaTensor(torch.randn(1, 10, 1, 10))
        result = SqueezeDim(dim=2)(t)
        self.assertEqual(result.spatial_ndim, result.affine.shape[-1] - 1)

    def test_splitdim_channel_dim_no_decrement(self):
        t = MetaTensor(torch.randn(3, 8, 7))
        for item in SplitDim(dim=0, keepdim=False)(t):
            if isinstance(item, MetaTensor):
                self.assertEqual(item.spatial_ndim, 1)

    def test_lazy_apply_pending_2d(self):
        """apply_pending uses spatial_ndim for 2D data instead of hardcoded 3."""
        from monai.transforms.lazy.functional import apply_pending
        from monai.utils.enums import LazyAttr

        t = MetaTensor(torch.randn(1, 10, 10), affine=torch.eye(3))
        self.assertEqual(t.spatial_ndim, 2)
        # Push a pending 2D affine operation
        pending_op = {
            LazyAttr.AFFINE: torch.eye(3, dtype=torch.float64),
            LazyAttr.SHAPE: (10, 10),
            LazyAttr.INTERP_MODE: "bilinear",
            LazyAttr.PADDING_MODE: "zeros",
        }
        t.push_pending_operation(pending_op)
        result, applied = apply_pending(t, overrides={"mode": "bilinear"})
        self.assertIsInstance(result, MetaTensor)
        self.assertEqual(len(applied), 1)

    def test_batch_slice_clamps_spatial_ndim(self):
        t = MetaTensor(torch.randn(10, 6, 5, 7), affine=torch.eye(4))
        t.is_batch = True
        t.meta["affine"] = torch.eye(4)[None].repeat(10, 1, 1)
        self.assertEqual(t.spatial_ndim, 3)
        sliced = t[0]
        self.assertEqual(sliced.shape, (6, 5, 7))
        self.assertEqual(sliced.spatial_ndim, 2)
        self.assertEqual(get_spatial_ndim(sliced), 2)

    def test_label_to_contour_batch_slice_2d(self):
        t = MetaTensor(torch.randint(0, 2, (10, 6, 5, 7)).float(), affine=torch.eye(4))
        t.is_batch = True
        t.meta["affine"] = torch.eye(4)[None].repeat(10, 1, 1)
        sliced = t[0]
        out = LabelToContour()(sliced)
        self.assertEqual(out.shape, sliced.shape)

    def test_rand_zoom_batch_slice_2d(self):
        t = MetaTensor(torch.randn(10, 1, 64, 64), affine=torch.eye(4))
        t.is_batch = True
        t.meta["affine"] = torch.eye(4)[None].repeat(10, 1, 1)
        sliced = t[0]
        zoom = RandZoom(prob=1.0, min_zoom=0.6, max_zoom=1.2)
        zoom.set_random_state(seed=0)
        zoom.randomize(sliced)
        self.assertEqual(len(zoom._zoom), 2)
        out = zoom(sliced)
        self.assertEqual(out.ndim, sliced.ndim)

    @skipUnless(has_einops, "Requires einops")
    def test_einops_rearrange_then_resize(self):
        """Reproduce the exact #6397 bug: einops.rearrange -> Resize."""
        from einops import rearrange

        x = MetaTensor(torch.randn(1, 1, 64, 64, 3))
        x.is_batch = True
        x.meta["affine"] = torch.eye(4)[None]
        x_ = rearrange(x, "b c h w d -> (b c) h w d")
        self.assertIsInstance(x_, MetaTensor)
        self.assertEqual(x_.spatial_ndim, 3)
        out = Resize(spatial_size=(32, 32, 3), mode="trilinear", align_corners=True)(x_)
        self.assertEqual(out.shape[-3:], (32, 32, 3))

    def test_affine_inverse_2d_metatensor(self):
        """Affine.inverse on 2D data: 4x4 affine with spatial_ndim=2."""
        img = MetaTensor(torch.randn(1, 32, 32), affine=torch.eye(4))
        self.assertEqual(img.spatial_ndim, 2)
        xform = Affine(rotate_params=(np.pi / 6,), padding_mode="zeros", image_only=True)
        result = xform(img)
        inv = xform.inverse(result)
        self.assertEqual(inv.shape, img.shape)
        self.assertEqual(len(inv.applied_operations), 0)

    def test_rotate_inverse_2d_metatensor(self):
        """Rotate.inverse on 2D data: 4x4 affine with spatial_ndim=2."""
        img = MetaTensor(torch.randn(1, 32, 32), affine=torch.eye(4))
        self.assertEqual(img.spatial_ndim, 2)
        xform = Rotate(angle=(np.pi / 4,), padding_mode="zeros")
        result = xform(img)
        inv = xform.inverse(result)
        self.assertEqual(inv.shape, img.shape)
        self.assertEqual(len(inv.applied_operations), 0)

    def test_rand_affine_inverse_2d_metatensor(self):
        """RandAffine.inverse on 2D data: 4x4 affine with spatial_ndim=2."""
        img = MetaTensor(torch.randn(1, 32, 32), affine=torch.eye(4))
        self.assertEqual(img.spatial_ndim, 2)
        xform = RandAffine(prob=1.0, rotate_range=(np.pi / 6,), padding_mode="zeros")
        xform.set_random_state(seed=42)
        result = xform(img)
        inv = xform.inverse(result)
        self.assertEqual(inv.shape, img.shape)
        self.assertEqual(len(inv.applied_operations), 0)


if __name__ == "__main__":
    unittest.main()
