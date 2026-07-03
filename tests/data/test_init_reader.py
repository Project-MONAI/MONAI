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

import os
import tempfile
import unittest

import numpy as np

from monai.data import ITKReader, NibabelReader, NrrdReader, NumpyReader, PILReader, PydicomReader
from monai.transforms import LoadImage, LoadImaged
from monai.utils import MetaKeys
from tests.test_utils import SkipIfNoModule


class TestInitLoadImage(unittest.TestCase):
    def test_load_image(self):
        instance1 = LoadImage(image_only=False, dtype=None)
        instance2 = LoadImage(image_only=True, dtype=None)
        self.assertIsInstance(instance1, LoadImage)
        self.assertIsInstance(instance2, LoadImage)

        for r in ["NibabelReader", "PILReader", "ITKReader", "NumpyReader", "NrrdReader", "PydicomReader", None]:
            inst = LoadImaged("image", reader=r)
            self.assertIsInstance(inst, LoadImaged)

    @SkipIfNoModule("nibabel")
    @SkipIfNoModule("cupy")
    @SkipIfNoModule("kvikio")
    def test_load_image_to_gpu(self):
        for to_gpu in [True, False]:
            instance1 = LoadImage(reader="NibabelReader", to_gpu=to_gpu)
            self.assertIsInstance(instance1, LoadImage)

            instance2 = LoadImaged("image", reader="NibabelReader", to_gpu=to_gpu)
            self.assertIsInstance(instance2, LoadImaged)

    @SkipIfNoModule("itk")
    @SkipIfNoModule("nibabel")
    @SkipIfNoModule("PIL")
    @SkipIfNoModule("nrrd")
    @SkipIfNoModule("pydicom")
    def test_readers(self):
        inst = ITKReader()
        self.assertIsInstance(inst, ITKReader)

        inst = NibabelReader()
        self.assertIsInstance(inst, NibabelReader)
        inst = NibabelReader(as_closest_canonical=True)
        self.assertIsInstance(inst, NibabelReader)

        inst = PydicomReader()
        self.assertIsInstance(inst, PydicomReader)

        inst = NumpyReader()
        self.assertIsInstance(inst, NumpyReader)
        inst = NumpyReader(npz_keys="test")
        self.assertIsInstance(inst, NumpyReader)

        inst = PILReader()
        self.assertIsInstance(inst, PILReader)

        inst = NrrdReader()
        self.assertIsInstance(inst, NrrdReader)

    @SkipIfNoModule("nibabel")
    @SkipIfNoModule("cupy")
    @SkipIfNoModule("kvikio")
    def test_readers_to_gpu(self):
        for to_gpu in [True, False]:
            inst = NibabelReader(to_gpu=to_gpu)
            self.assertIsInstance(inst, NibabelReader)

    @SkipIfNoModule("nibabel")
    def test_nibabel_reader_avoids_eager_c_order_copy(self):
        import nibabel as nib

        test_image = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as tempdir:
            for suffix in (".nii", ".nii.gz"):
                with self.subTest(suffix=suffix):
                    filename = os.path.join(tempdir, f"test_image{suffix}")
                    nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)

                    reader = NibabelReader(mmap=False)
                    img = reader.read(filename)
                    data, _ = reader.get_data(img)

                    np.testing.assert_array_equal(data, test_image)
                    # The reader must not force an eager C-order copy; the native
                    # (F-order) layout from nibabel should be preserved here.
                    self.assertFalse(data.flags.c_contiguous)

    @SkipIfNoModule("pydicom")
    def test_pydicom_reader_get_affine_single_slice_with_last_position(self):
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200032": {"Value": [10.0, 20.0, 30.0]},
            "00280030": {"Value": [0.5, 0.25]},
            "lastImagePositionPatient": np.array([10.0, 20.0, 30.0]),
            MetaKeys.SPATIAL_SHAPE: np.array([64, 64, 1]),
        }

        affine = reader._get_affine(metadata, lps_to_ras=False)

        np.testing.assert_allclose(affine[0, 2], 0.0)
        np.testing.assert_allclose(affine[1, 2], 0.0)
        np.testing.assert_allclose(affine[2, 2], 1.0)

    @SkipIfNoModule("pydicom")
    def test_pydicom_reader_get_affine_multi_slice_uses_last_position(self):
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200032": {"Value": [0.0, 0.0, 0.0]},
            "00280030": {"Value": [1.0, 1.0]},
            "lastImagePositionPatient": np.array([0.0, 0.0, 8.0]),
            MetaKeys.SPATIAL_SHAPE: np.array([8, 8, 5]),
        }

        affine = reader._get_affine(metadata, lps_to_ras=False)

        np.testing.assert_allclose(affine[0, 2], 0.0)
        np.testing.assert_allclose(affine[1, 2], 0.0)
        np.testing.assert_allclose(affine[2, 2], 2.0)


if __name__ == "__main__":
    unittest.main()
