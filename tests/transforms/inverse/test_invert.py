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

import sys
import unittest
from copy import deepcopy

import numpy as np
import torch

from monai.data import DataLoader, Dataset, MetaTensor, create_test_image_3d, decollate_batch
from monai.transforms import (
    CastToType,
    Compose,
    EnsureChannelFirst,
    Invert,
    Lambda,
    LoadImage,
    Orientation,
    RandAffine,
    RandAxisFlip,
    RandFlip,
    RandRotate,
    RandRotate90,
    RandZoom,
    ResizeWithPadOrCrop,
    Spacing,
    convert_applied_interp_mode,
)
from monai.utils import optional_import, set_determinism
from tests.test_utils import assert_allclose, make_nifti_image


class TestInvert(unittest.TestCase):
    def test_invert(self):
        set_determinism(seed=0)
        im_fname = make_nifti_image(create_test_image_3d(101, 100, 107, noise_max=100)[1])  # label image, discrete
        data = [im_fname for _ in range(12)]
        transform = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation("RPS"),
                Spacing(pixdim=(1.2, 1.01, 0.9), mode=1, dtype=np.float32),
                RandFlip(prob=0.5, spatial_axis=[1, 2]),
                RandAxisFlip(prob=0.5),
                RandRotate90(prob=0, spatial_axes=(1, 2)),
                RandZoom(prob=0.5, min_zoom=0.5, max_zoom=1.1, keep_size=True),
                RandRotate(prob=0.5, range_x=np.pi, mode="bilinear", align_corners=True, dtype=np.float64),
                RandAffine(prob=0.5, rotate_range=np.pi, mode="nearest"),
                ResizeWithPadOrCrop(100),
                CastToType(dtype=torch.uint8),
            ]
        )

        # num workers = 0 for mac or gpu transforms
        num_workers = 0 if sys.platform != "linux" or torch.cuda.is_available() else 2
        dataset = Dataset(data, transform=transform)
        self.assertIsInstance(transform.inverse(dataset[0]), MetaTensor)
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=1)
        inverter = Invert(transform=transform, nearest_interp=True, device="cpu", post_func=torch.as_tensor)

        for d in loader:
            d = decollate_batch(d)
            for item in d:
                orig = deepcopy(item)
                i = inverter(item)
                self.assertTupleEqual(orig.shape[1:], (100, 100, 100))
                # check the nearest interpolation mode
                assert_allclose(i.to(torch.uint8).to(torch.float), i.to(torch.float))
                self.assertTupleEqual(i.shape[1:], (101, 100, 107))
        # check labels match
        reverted = i.detach().cpu().numpy().astype(np.int32)
        original = LoadImage(image_only=True)(data[-1])
        n_good = np.sum(np.isclose(reverted, original.numpy(), atol=1e-3))
        reverted_name = i.meta["filename_or_obj"]
        original_name = original.meta["filename_or_obj"]
        self.assertEqual(reverted_name, original_name)
        print("invert diff", reverted.size - n_good)
        self.assertTrue((reverted.size - n_good) < 300000, f"diff. {reverted.size - n_good}")
        set_determinism(seed=None)

    def test_invert_warn_pending(self):
        # this test shouldn't raise a warning or error any more as that issue was fixed
        # by https://github.com/Project-MONAI/MONAI/pull/6257
        set_determinism(seed=0)
        im_fname = make_nifti_image(create_test_image_3d(101, 100, 107, noise_max=100)[1])  # label image, discrete
        transform = Compose(
            [LoadImage(image_only=True), EnsureChannelFirst(), Orientation("RPS"), Lambda(func=lambda x: x)], lazy=True
        )
        output = transform([im_fname for _ in range(2)])
        transform.inverse(output)


class TestInvertInterp(unittest.TestCase):
    """Tests for the optional `interp` argument of `Invert` (issue #7836)."""

    def _make_edge_soft_seg(self, shape):
        axes = [torch.arange(float(s)) for s in shape]
        grids = torch.meshgrid(*axes, indexing="ij")
        center = tuple(s / 2.0 for s in shape)
        r = torch.sqrt(sum((g - c) ** 2 for g, c in zip(grids, center)))
        field = torch.sigmoid((min(shape) * 0.3 - r) * 1.8).clamp(0.0, 1.0)
        return MetaTensor(field.unsqueeze(0), affine=torch.eye(len(shape) + 1))

    def test_positional_arguments_unchanged(self):
        set_determinism(seed=0)
        fwd = Compose([Spacing(pixdim=1.4, mode="bilinear")])
        # legacy positional order: transform, nearest_interp, device, post_func, to_tensor
        inverter = Invert(fwd, False, None, None, True)
        self.assertFalse(inverter.nearest_interp)
        self.assertIsNone(inverter.device)
        self.assertIsNone(inverter.post_func)
        self.assertTrue(inverter.to_tensor)
        # the appended parameter defaults to None and does not shift any positional argument
        self.assertIsNone(inverter.interp)

    def test_invalid_interp_rejected(self):
        set_determinism(seed=0)
        fwd = Compose([Spacing(pixdim=1.4, mode="bilinear")])
        with self.assertRaises(ValueError):
            Invert(transform=fwd, interp="banana")
        with self.assertRaises(ValueError):
            Invert(transform=fwd, interp=9)
        with self.assertRaises((TypeError, ValueError)):
            Invert(transform=fwd, interp=True)

    def test_interp(self):
        set_determinism(seed=0)
        data = self._make_edge_soft_seg((16, 20, 20))
        fwd = Compose([Spacing(pixdim=1.4, mode="bilinear")])
        resampled = fwd(data)
        out_default = Invert(transform=fwd, to_tensor=False)(resampled.clone())
        out_interp = Invert(transform=fwd, interp="bilinear", to_tensor=False)(resampled.clone())
        # an explicitly supplied `interp` takes precedence over the `nearest_interp=True` default
        self.assertFalse(torch.allclose(out_default, out_interp))
        # for this saturated, in-range input the bilinear inversion stays inside [0, 1]
        # (a convex combination of in-range values -- data-specific, not a general guarantee)
        self.assertGreaterEqual(float(out_interp.min()), -1e-6)
        self.assertLessEqual(float(out_interp.max()), 1.0 + 1e-6)
        # and matches the manual recorded-mode rewrite mechanism
        sim = resampled.clone()
        sim.applied_operations = convert_applied_interp_mode(sim.applied_operations, mode="bilinear")
        expected = fwd.inverse(sim)
        assert_allclose(out_interp, expected, type_test=False)
        # transformation metadata is preserved: the affine is restored and the trace fully consumed
        self.assertIsInstance(out_interp, MetaTensor)
        assert_allclose(out_interp.affine, data.affine, type_test=False)
        self.assertEqual(out_interp.applied_operations, [])

    def test_legacy_default_unchanged(self):
        set_determinism(seed=0)
        data = self._make_edge_soft_seg((16, 20, 20))
        fwd = Compose([Spacing(pixdim=1.4, mode="bilinear")])
        resampled = fwd(data)
        out_default = Invert(transform=fwd, to_tensor=False)(resampled.clone())
        sim = resampled.clone()
        sim.applied_operations = convert_applied_interp_mode(sim.applied_operations, mode="nearest", align_corners=None)
        expected = fwd.inverse(sim)
        assert_allclose(out_default, expected, type_test=False)

    @unittest.skipUnless(optional_import("scipy")[1], "Requires SciPy.")
    def test_recorded_spline_reuse(self):
        set_determinism(seed=0)
        data = self._make_edge_soft_seg((16, 20, 20))
        fwd = Compose([Spacing(pixdim=1.4, mode=2)])
        resampled = fwd(data)
        out_recorded = Invert(transform=fwd, nearest_interp=False, to_tensor=False)(resampled.clone())
        expected = fwd.inverse(resampled.clone())
        assert_allclose(out_recorded, expected, type_test=False)
        out_nearest = Invert(transform=fwd, to_tensor=False)(resampled.clone())
        self.assertFalse(torch.allclose(out_recorded, out_nearest))

    @unittest.skipUnless(optional_import("scipy")[1], "Requires SciPy.")
    def test_interp_spline_order(self):
        set_determinism(seed=0)
        data = self._make_edge_soft_seg((16, 20, 20))
        fwd = Compose([Spacing(pixdim=1.4, mode="bilinear")])
        resampled = fwd(data)
        sim = resampled.clone()
        sim.applied_operations = convert_applied_interp_mode(sim.applied_operations, mode=2)
        expected = fwd.inverse(sim)
        out = Invert(transform=fwd, interp=2, to_tensor=False)(resampled.clone())
        assert_allclose(out, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
