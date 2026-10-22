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

import numpy as np
import torch
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import CLAHE
from monai.transforms.utils import clahe
from monai.utils import get_equivalent_dtype, optional_import
from tests.test_utils import TEST_NDARRAYS, SkipIfNoModule, assert_allclose, skip_if_no_cuda

sitk, has_sitk = optional_import("SimpleITK")


def _img_2d(seed=0):
    return np.random.default_rng(seed).random((1, 24, 24)).astype(np.float32)


def _brain(size=16, seed=0):
    """A brain-like volume plus its tissue mask, background exactly zero."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    c = (size - 1) / 2
    tissue = (((xx - c) / (size * 0.34)) ** 2 + ((yy - c) / (size * 0.44)) ** 2 + ((zz - c) / (size * 0.38)) ** 2) <= 1
    vol = np.zeros((1, size, size, size), np.float32)
    vol[0][tissue] = 300.0 + rng.normal(0, 40, tissue.sum())
    return vol, tissue


def _reference(channel, radius=5, alpha=0.3, beta=0.3):
    """Call SimpleITK directly, reversing the radius for its x-first axis order."""
    r = [radius] * channel.ndim if isinstance(radius, int) else list(radius)
    return sitk.GetArrayFromImage(
        sitk.AdaptiveHistogramEqualization(
            sitk.GetImageFromArray(channel.astype(np.float32)), radius=list(reversed(r)), alpha=alpha, beta=beta
        )
    )


@SkipIfNoModule("SimpleITK")
class TestCLAHE(unittest.TestCase):
    def test_matches_simpleitk(self):
        img = _img_2d()
        assert_allclose(CLAHE()(img)[0], _reference(img[0]), type_test=False, rtol=1e-5, atol=1e-5)

    @parameterized.expand([({},), ({"radius": 3},), ({"alpha": 0.0, "beta": 0.0},), ({"alpha": 1.0, "beta": 0.5},)])
    def test_matches_simpleitk_arguments(self, arguments):
        img = _img_2d()
        expected = _reference(img[0], **arguments)
        assert_allclose(CLAHE(**arguments)(img)[0], expected, type_test=False, rtol=1e-5, atol=1e-5)

    def test_alpha_beta_one_is_pass_through(self):
        """ITK documents alpha=1, beta=1 as returning the input unchanged."""
        img = _img_2d()
        assert_allclose(CLAHE(alpha=1.0, beta=1.0)(img)[0], img[0], type_test=False, atol=1e-5)

    # --- the axis-order trap --------------------------------------------------------------

    def test_anisotropic_radius_lands_on_the_right_axes(self):
        """SimpleITK indexes x-first, so an unreversed radius would smooth the wrong axes."""
        vol = np.random.default_rng(0).random((1, 8, 16, 32)).astype(np.float32)
        radius = (1, 3, 7)
        out = CLAHE(radius=radius)(vol)
        self.assertTupleEqual(tuple(out.shape), (1, 8, 16, 32))
        assert_allclose(out[0], _reference(vol[0], radius=radius), type_test=False, rtol=1e-5, atol=1e-5)

    def test_transpose_equivariance(self):
        """Transposing the volume and its radius together must give the transposed result."""
        vol = np.random.default_rng(0).random((1, 6, 10, 14)).astype(np.float32)
        radius = (1, 3, 5)
        direct = CLAHE(radius=radius)(vol).numpy()[0]
        perm = (2, 0, 1)
        moved = CLAHE(radius=tuple(radius[i] for i in perm))(np.transpose(vol, (0, *(p + 1 for p in perm)))).numpy()[0]
        assert_allclose(np.transpose(moved, np.argsort(perm)), direct, type_test=False, rtol=1e-5, atol=1e-5)

    # --- 2D / 3D / channels ---------------------------------------------------------------

    @parameterized.expand([((1, 12, 12),), ((1, 10, 12, 8),), ((3, 12, 12),), ((4, 8, 8, 8),)])
    def test_shapes_preserved(self, shape):
        vol = np.random.default_rng(0).random(shape).astype(np.float32)
        self.assertTupleEqual(tuple(CLAHE(radius=2)(vol).shape), shape)

    def test_channels_are_independent(self):
        a = np.random.default_rng(0).random((1, 12, 12)).astype(np.float32)
        b = np.random.default_rng(1).random((1, 12, 12)).astype(np.float32) * 1000.0
        stacked = CLAHE()(np.concatenate([a, b]))
        assert_allclose(stacked[0], CLAHE()(a)[0], type_test=False, rtol=1e-5, atol=1e-5)

    # --- range, masking -------------------------------------------------------------------

    def test_keeps_input_range(self):
        """The filter works on the intensities as given, with no rescaling to [0, 1]."""
        img = np.random.default_rng(0).uniform(-1024.0, 3000.0, (1, 16, 16)).astype(np.float32)
        out = CLAHE()(img)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertGreater(float(out.min()), -2000.0)
        self.assertLess(float(out.max()), 4000.0)

    def test_mask_hands_back_excluded_voxels(self):
        vol, tissue = _brain()
        out = CLAHE(mask=tissue)(vol).numpy()[0]
        assert_allclose(out[~tissue], vol[0][~tissue], type_test=False)

    def test_nonzero_hands_back_background(self):
        vol, tissue = _brain()
        out = CLAHE(nonzero=True)(vol).numpy()[0]
        assert_allclose(out[~tissue], np.zeros(int((~tissue).sum()), np.float32), type_test=False)

    def test_mask_at_call_time(self):
        vol, tissue = _brain()
        assert_allclose(CLAHE()(vol, tissue), CLAHE(mask=tissue)(vol), type_test=False, rtol=1e-5, atol=1e-5)

    def test_spatial_mask_broadcast_over_channels(self):
        vol, tissue = _brain()
        out = CLAHE(mask=tissue)(np.concatenate([vol, vol]))
        self.assertTupleEqual(tuple(out.shape), (2, 16, 16, 16))
        assert_allclose(out[0], out[1], type_test=False, rtol=1e-5, atol=1e-5)

    def test_mask_shape_mismatch(self):
        with self.assertRaises(ValueError):
            CLAHE(mask=np.ones((8, 8), bool))(_img_2d())

    def test_empty_selection(self):
        """An all-False mask on a real image is caller error, not a silent no-op."""
        with self.assertRaises(ValueError):
            CLAHE(mask=np.zeros((24, 24), bool))(_img_2d())

    def test_nonzero_on_flat_channel_is_not_an_empty_selection(self):
        """`nonzero` empties the selection on a flat channel, but the constant guard runs first."""
        flat = np.zeros((1, 10, 10), np.float32)
        assert_allclose(CLAHE(nonzero=True)(flat)[0], flat[0], type_test=False)
        vol = np.zeros((2, 10, 10, 10), np.float32)
        vol[0] = np.random.default_rng(0).random((10, 10, 10))
        out = CLAHE(nonzero=True)(vol)
        self.assertTrue(bool(torch.isfinite(out).all()))
        assert_allclose(out[1], vol[1], type_test=False)

    # --- containers, dtype, metadata ------------------------------------------------------

    @parameterized.expand(TEST_NDARRAYS)
    def test_container_agnostic(self, im_type):
        img = _img_2d()
        out = CLAHE()(im_type(img))
        self.assertIsInstance(out, MetaTensor)
        assert_allclose(out[0], _reference(img[0]), type_test=False, rtol=1e-5, atol=1e-5)

    @parameterized.expand([(np.float32,), (np.float64,), (None,)])
    def test_dtype(self, dtype):
        out = CLAHE(dtype=dtype)(_img_2d())
        self.assertEqual(
            get_equivalent_dtype(out.dtype, data_type=np.ndarray), dtype if dtype is not None else np.float32
        )

    def test_meta_preserved(self):
        img = MetaTensor(torch.as_tensor(_img_2d()), meta={"filename_or_obj": "scan.nii.gz"})
        self.assertEqual(CLAHE()(img).meta["filename_or_obj"], "scan.nii.gz")

    # --- constant input: SimpleITK returns all-NaN for it on its own -------------------

    @parameterized.expand([(0.0,), (0.5,), (7.0,), (-300.0,)])
    def test_constant_image(self, value):
        """A constant image has no contrast to redistribute; the backend would give NaNs."""
        img = np.full((1, 16, 16), value, dtype=np.float32)
        out = CLAHE()(img)
        self.assertTrue(bool(torch.isfinite(out).all()))
        assert_allclose(out[0], img[0], type_test=False)

    def test_constant_channel_does_not_poison_the_sample(self):
        """One empty modality must not corrupt the channels beside it."""
        vol = np.zeros((2, 12, 12, 12), np.float32)
        vol[0] = np.random.default_rng(0).random((12, 12, 12))
        out = CLAHE()(vol)
        self.assertTrue(bool(torch.isfinite(out).all()))
        assert_allclose(out[1], vol[1], type_test=False)
        assert_allclose(out[0], CLAHE()(vol[:1])[0], type_test=False, rtol=1e-5, atol=1e-5)

    def test_constant_image_does_not_alias_input(self):
        src = np.full((8, 8), 3.0, np.float32)
        out = clahe(src)
        self.assertIsNot(out, src)
        out[0, 0] = 999.0
        assert_allclose(src, np.full((8, 8), 3.0, np.float32), type_test=False)

    # --- thread control -----------------------------------------------------------------

    @parameterized.expand([(1,), (2,)])
    def test_num_threads_does_not_change_the_result(self, n):
        """`num_threads` caps SimpleITK's thread use without altering the output."""
        img = _img_2d()
        assert_allclose(CLAHE(num_threads=n)(img), CLAHE()(img), type_test=False, rtol=1e-5, atol=1e-5)

    # --- failure modes --------------------------------------------------------------------

    @parameterized.expand([(np.nan,), (np.inf,), (-np.inf,)])
    def test_non_finite_input(self, bad):
        """SimpleITK segfaults the process on NaN, so this must be caught before the call."""
        img = _img_2d()
        img[0, 0, 0] = bad
        with self.assertRaises(ValueError):
            CLAHE()(img)

    @parameterized.expand([(-0.1,), (1.5,)])
    def test_invalid_alpha_fails_at_construction(self, alpha):
        with self.assertRaises(ValueError):
            CLAHE(alpha=alpha)

    @parameterized.expand([(-0.1,), (1.5,)])
    def test_invalid_beta_fails_at_construction(self, beta):
        with self.assertRaises(ValueError):
            CLAHE(beta=beta)

    @parameterized.expand([(0,), (-1,), (5.7,), (None,), ((2, 2, 2),)])
    def test_invalid_radius(self, radius):
        """`radius` must be whole numbers of at least 1, matching the image dimensions."""
        with self.assertRaises(ValueError):
            CLAHE(radius=radius)(_img_2d())

    @parameterized.expand([(5,), (5.0,)])
    def test_whole_number_radius_accepted(self, radius):
        self.assertTupleEqual(tuple(CLAHE(radius=radius)(_img_2d()).shape), (1, 24, 24))

    @parameterized.expand([(0,), (-1,), (2.5,)])
    def test_invalid_num_threads_fails_at_construction(self, num_threads):
        """Validated eagerly, like `alpha` and `beta`, so a bad pipeline fails before data loads."""
        with self.assertRaises(ValueError):
            CLAHE(num_threads=num_threads)

    def test_float64_precision_is_kept(self):
        """SimpleITK equalizes float64 natively, so double precision is not thrown away."""
        img = np.random.default_rng(0).random((1, 16, 16)).astype(np.float64)
        self.assertEqual(clahe(img[0]).dtype, np.float64)
        self.assertEqual(clahe(img[0].astype(np.float32)).dtype, np.float32)

    def test_requires_channel_dim(self):
        with self.assertRaises(ValueError) as cm:
            CLAHE()(np.random.default_rng(0).random((16, 16)).astype(np.float32))
        self.assertIn("(16, 16)", str(cm.exception))

    def test_empty_image(self):
        with self.assertRaises(ValueError):
            CLAHE()(np.zeros((1, 0, 0), np.float32))

    # --- CUDA round-trip: the filter is CPU-only, so the device must survive it ----------

    @skip_if_no_cuda
    def test_cuda_input_returns_to_its_device(self):
        """SimpleITK runs on CPU, so a CUDA tensor must come back on CUDA with equal values."""
        img = _img_2d()
        expected = CLAHE()(img)
        out = CLAHE()(torch.as_tensor(img, device="cuda"))
        self.assertEqual(out.device.type, "cuda")
        assert_allclose(out.cpu(), expected.cpu(), type_test=False, rtol=1e-5, atol=1e-5)

    @skip_if_no_cuda
    def test_cuda_metatensor_keeps_device_and_meta(self):
        img = MetaTensor(torch.as_tensor(_img_2d()), meta={"filename_or_obj": "scan.nii.gz"}).to("cuda")
        out = CLAHE()(img)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.meta["filename_or_obj"], "scan.nii.gz")

    @skip_if_no_cuda
    def test_cuda_mask_on_device(self):
        """A mask handed in as a CUDA tensor must be accepted alongside a CUDA image."""
        vol, tissue = _brain()
        out = CLAHE()(torch.as_tensor(vol, device="cuda"), torch.as_tensor(tissue, device="cuda"))
        self.assertEqual(out.device.type, "cuda")
        assert_allclose(out.cpu(), CLAHE(mask=tissue)(vol).cpu(), type_test=False, rtol=1e-5, atol=1e-5)

    # --- functional helper ----------------------------------------------------------------

    def test_functional_helper(self):
        img = _img_2d()
        assert_allclose(clahe(img[0]), _reference(img[0]), type_test=False, rtol=1e-5, atol=1e-5)

    def test_functional_helper_rejects_1d(self):
        with self.assertRaises(ValueError):
            clahe(np.random.default_rng(0).random(10).astype(np.float32))


if __name__ == "__main__":
    unittest.main()
