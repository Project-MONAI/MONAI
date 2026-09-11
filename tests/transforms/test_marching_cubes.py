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
from monai.transforms import MarchingCubes, MarchingCubesd
from monai.transforms.utils import get_marching_cubes_surface
from monai.utils import optional_import
from tests.test_utils import TEST_NDARRAYS, SkipIfNoModule

measure, has_measure = optional_import("skimage.measure")


def _cube(channel_first=True):
    vol = np.zeros((10, 10, 10), np.float32)
    vol[3:7, 3:7, 3:7] = 1.0
    return vol[np.newaxis] if channel_first else vol


@SkipIfNoModule("skimage.measure")
class TestMarchingCubes(unittest.TestCase):
    def test_matches_skimage(self):
        vol = _cube()
        verts, faces = MarchingCubes()(vol)
        e_verts, e_faces, _, _ = measure.marching_cubes(vol[0], level=0.5)
        self.assertTupleEqual(verts.shape, e_verts.shape)
        self.assertTupleEqual(faces.shape, e_faces.shape)
        np.testing.assert_allclose(verts, e_verts, rtol=1e-5)
        np.testing.assert_array_equal(faces, e_faces)
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)

    @parameterized.expand(TEST_NDARRAYS)
    def test_input_types(self, im_type):
        vol = im_type(_cube())
        verts, faces = MarchingCubes()(vol)
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertTrue((faces.max(0) < len(verts)).all())

    def test_spacing_scales_vertices(self):
        vol = _cube()
        verts, _ = MarchingCubes()(vol)
        verts2, _ = MarchingCubes(spacing=2.0)(vol)
        np.testing.assert_allclose(verts2, verts * 2.0, rtol=1e-5)

    def test_metatensor_pixdim_spacing(self):
        vol = _cube()
        affine = torch.diag(torch.as_tensor([2.0, 3.0, 4.0, 1.0]))
        verts, _ = MarchingCubes()(MetaTensor(vol, affine=affine))
        verts_ref, _ = MarchingCubes(spacing=(2.0, 3.0, 4.0))(vol)
        np.testing.assert_allclose(verts, verts_ref, rtol=1e-5)

    def test_multi_channel_returns_list(self):
        vol = np.concatenate([_cube(), _cube()], axis=0)
        out = MarchingCubes()(vol)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        for verts, _faces in out:
            self.assertEqual(verts.shape[1], 3)

    def test_return_normals_values(self):
        verts, faces, normals, values = MarchingCubes(return_normals_values=True)(_cube())
        self.assertEqual(normals.shape, verts.shape)
        self.assertEqual(len(values), len(verts))

    def test_step_size(self):
        verts, faces = MarchingCubes(step_size=2)(_cube())
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)

    def test_dict_wrapper(self):
        vol = _cube()
        out = MarchingCubesd(keys=["seg"])({"seg": vol})["seg"]
        verts, _ = MarchingCubes()(vol)
        np.testing.assert_allclose(out[0], verts, rtol=1e-5)
        # allow missing keys
        out = MarchingCubesd(keys=["missing"], allow_missing_keys=True)({"seg": vol})
        self.assertIn("seg", out)

    def test_util_matches_skimage(self):
        vol = _cube(channel_first=False)
        verts, faces, _, _ = get_marching_cubes_surface(vol, level=0.5)
        e_verts, e_faces, _, _ = measure.marching_cubes(vol, level=0.5)
        np.testing.assert_allclose(verts, e_verts, rtol=1e-5)
        np.testing.assert_array_equal(faces, e_faces)

    def test_errors(self):
        with self.assertRaises(ValueError):
            MarchingCubes()(np.zeros((1, 10, 10), np.float32))
        with self.assertRaises(ValueError):
            # empty channel axis
            MarchingCubes()(np.zeros((0, 10, 10, 10), np.float32))
        with self.assertRaises(ValueError):
            get_marching_cubes_surface(np.zeros((10, 10), np.float32))
        with self.assertRaises(ValueError):
            get_marching_cubes_surface(np.zeros((10, 10, 10), np.float32), mask=np.ones((5, 5, 5), bool))
        with self.assertRaises(ValueError):
            # empty volume has no isosurface
            MarchingCubes()(np.zeros((1, 10, 10, 10), np.float32))


if __name__ == "__main__":
    unittest.main()
