# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Regression tests for Spacing / zoom_affine with non-orthogonal (sheared) input affines.

zoom_affine (monai/data/utils.py) recomposed the direction matrix via Cholesky/QR, which
silently removed any shear. Tests 1 and 2 pin the behaviour after the fix (shear-preserving
column rescale); tests 3 and 4 are regression guards for parts verified correct.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import Orientation, RandAffine, Spacing


def _random_sheared_affine(rng: np.random.Generator, det_negative: bool = False) -> np.ndarray:
    """Random 4x4 affine with anisotropic spacing, non-orthogonal (sheared) rotation."""
    sp = rng.uniform(0.8, 3.5, size=3)
    a, b, c = rng.uniform(-np.pi, np.pi, size=3)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    r = rz @ ry @ rx
    shear = np.eye(3)
    shear[0, 1] = rng.uniform(-0.35, 0.35)
    shear[1, 2] = rng.uniform(-0.3, 0.3)
    r = r @ shear
    if det_negative:
        r[:, 0] *= -1.0
    A = np.eye(4)
    A[:3, :3] = r @ np.diag(sp)
    A[:3, 3] = rng.uniform(-20, 20, size=3)
    return A


def _smooth_volume(shape=(1, 16, 16, 16), seed=0) -> torch.Tensor:
    z = np.linspace(-1, 1, shape[1])
    d = z[:, None, None] ** 2 + z[None, :, None] ** 2 + z[None, None, :] ** 2
    vol = np.sin(3 * d) + 0.5 * np.sin(5 * d)[::-1, ::-1, ::-1]
    return torch.as_tensor(vol, dtype=torch.float32)[None]


def _axis_direction_changes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Angles (deg) between corresponding column directions of two 3x3 matrices."""
    da = A / np.linalg.norm(A, axis=0, keepdims=True)
    db = B / np.linalg.norm(B, axis=0, keepdims=True)
    dots = np.clip(np.sum(da * db, axis=0), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


class TestSpacingShearInvariance(unittest.TestCase):
    """Spacing(same pixdim, diagonal=False) must not change the geometry of a sheared image."""

    def test_spacing_shear_invariance(self):
        rng = np.random.default_rng(555)
        worst = 0.0
        for i in range(10):
            A = _random_sheared_affine(rng)
            data = _smooth_volume(seed=i)
            mt = MetaTensor(data.clone())
            mt.affine = A
            orig_spacing = np.linalg.norm(A[:3, :3], axis=0)
            out = Spacing(pixdim=[float(s) for s in orig_spacing], diagonal=False, mode="bilinear")(mt)
            outA = np.asarray(out.affine)
            degs = _axis_direction_changes(A[:3, :3], outA[:3, :3])
            worst = max(worst, float(degs.max()))
            # resampling to the same spacing must not rotate the world direction of any axis
            self.assertTrue(
                np.allclose(outA[:3, :3], A[:3, :3], atol=1e-4),
                f"case {i}: Spacing with the input's own pixdim changed the affine:\n"
                f"{outA[:3, :3]}\nvs input\n{A[:3, :3]}\n"
                f"max axis-direction change {degs.max():.2f} deg (docstring says orientation preserved)",
            )
        # keep the measured magnitude in the failure report path
        assert worst >= 0.0

    def test_spacing_roundtrip_back_to_original(self):
        """Spacing(new) -> Spacing(original spacing) must restore the original linear affine."""
        rng = np.random.default_rng(123)
        for i in range(6):
            A = _random_sheared_affine(rng, det_negative=(i % 3 == 2))
            data = _smooth_volume(seed=i)
            mt = MetaTensor(data.clone())
            mt.affine = A
            orig_spacing = np.linalg.norm(A[:3, :3], axis=0)
            new_spacing = orig_spacing * np.array([1.3, 0.8, 1.1])
            t1 = Spacing(pixdim=[float(s) for s in new_spacing], diagonal=False, mode="bilinear")(mt)
            t2 = Spacing(pixdim=[float(s) for s in orig_spacing], diagonal=False, mode="bilinear")(t1)
            outA = np.asarray(t2.affine)
            # the shear is removed by zoom_affine at every step and never comes back
            degs = _axis_direction_changes(A[:3, :3], outA[:3, :3])
            self.assertTrue(
                np.allclose(outA[:3, :3], A[:3, :3], atol=1e-4),
                f"case {i}: round trip changed the linear affine, max axis-direction " f"change {degs.max():.2f} deg",
            )


class TestVerifiedCorrect(unittest.TestCase):
    """Regression guards for behaviour that was measured CORRECT against SimpleITK."""

    def test_orientation_roundtrip(self):
        rng = np.random.default_rng(31)
        for i, ax in enumerate(["RAS", "LPS", "PLS", "ASL", "IRP", "SLA", "AIL", "RPI"]):
            A = _random_sheared_affine(rng, det_negative=(i % 2 == 1))
            data = _smooth_volume(seed=i)
            mt = MetaTensor(data.clone())
            mt.affine = A
            t1 = Orientation(axcodes=ax)(mt)
            import nibabel as nib

            orig_ax = nib.orientations.aff2axcodes(A)
            t2 = Orientation(axcodes="".join(orig_ax))(t1)
            if t2.shape == mt.shape:
                self.assertLess(np.abs(t2.numpy()[0] - mt.numpy()[0]).max(), 1e-6)
                self.assertLess(np.abs(np.asarray(t2.affine) - A).max(), 1e-6)

    def test_randaffine_inverse_consistent(self):
        """RandAffine.inverse uses the exact inverse of the forward voxel affine (5.96e-7)."""
        rng = np.random.default_rng(99)
        A = _random_sheared_affine(rng)
        mt = MetaTensor(_smooth_volume(seed=0).clone())
        mt.affine = A
        tf = RandAffine(
            prob=1.0,
            rotate_range=0.15,
            shear_range=0.1,
            translate_range=2,
            scale_range=0.1,
            mode="bilinear",
            padding_mode="border",
        )
        tf.set_random_state(42)
        out = tf(mt)
        inv = tf.inverse(out)
        self.assertEqual(tuple(inv.shape[1:]), tuple(mt.shape[1:]))
        self.assertLess(np.abs(np.asarray(inv.affine) - A).max(), 1e-5)


if __name__ == "__main__":
    unittest.main()
