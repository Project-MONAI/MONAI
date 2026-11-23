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
from parameterized import parameterized

from monai.data import ITKReader, NibabelReader, NrrdReader
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage
from monai.utils import optional_import
from tests.test_utils import SkipIfNoModule

nib, has_nib = optional_import("nibabel")
nrrd, has_nrrd = optional_import("nrrd")
itk, has_itk = optional_import("itk", allow_namespace_pkg=True)

NOT_FINITE = "not finite|non-finite"


def _write_nifti(path: str, affine: np.ndarray) -> str:
    img = nib.Nifti1Image(np.ones((4, 4, 4), np.float32), affine)
    img.set_qform(None, code=0)
    img.set_sform(affine, code=1)
    nib.save(img, path)
    return path


def _write_nrrd(path: str, directions: str, origin: str) -> str:
    header = (
        "NRRD0004\ntype: float\ndimension: 3\nspace: left-posterior-superior\nsizes: 4 4 4\n"
        f"space directions: {directions}\nendian: little\nencoding: raw\nspace origin: {origin}\n\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.ones(64, np.float32).tobytes())
    return path


class _NanAffineReader(ImageReader):
    """A user-defined reader that returns a non-finite affine."""

    def verify_suffix(self, filename):
        return True

    def read(self, data, **kwargs):
        return data

    def get_data(self, img):
        affine = np.eye(4)
        affine[0, 3] = np.nan
        return np.ones((4, 4, 4), np.float32), {"affine": affine, "original_affine": np.eye(4)}


class TestReaderAffineFinite(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = self._dir.name

    @SkipIfNoModule("nibabel")
    def test_nifti_nan_translation(self):
        # nan only in the translation column: correct_nifti_header_if_necessary leaves it unchanged
        affine = np.eye(4)
        affine[0, 3] = np.nan
        path = _write_nifti(os.path.join(self.dir, "nan.nii"), affine)
        reader = NibabelReader()
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            reader.get_data(reader.read(path))
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            LoadImage(image_only=True)(path)

    @SkipIfNoModule("nibabel")
    def test_nifti_inf_high_dimensional(self):
        # correct_nifti_header_if_necessary does not check images with more than 4 dimensions;
        # the header is written directly because nibabel will not build an image from this affine
        shape = (4, 4, 4, 1, 2)
        header = nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4)).header.copy()
        header.set_qform(None, code=0)
        header["sform_code"] = 1
        header["srow_x"] = [np.inf, 0, 0, 0]
        path = os.path.join(self.dir, "inf5d.nii")
        nib.save(nib.Nifti1Image(np.ones(shape, np.float32), None, header=header), path)
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            LoadImage(image_only=True)(path)

    @SkipIfNoModule("nibabel")
    def test_nifti_finite_loads(self):
        affine = np.diag([2.0, 1.5, 1.0, 1.0])
        affine[:3, 3] = [10.0, -5.0, 3.0]
        path = _write_nifti(os.path.join(self.dir, "ok.nii"), affine)
        img = LoadImage(image_only=True)(path)
        np.testing.assert_allclose(np.asarray(img.affine), affine)

    @parameterized.expand(
        [
            ("inf_direction_nan_origin", "(inf,0,0) (0,1,0) (0,0,1)", "(nan,0,0)"),
            ("inf_direction", "(inf,0,0) (0,1,0) (0,0,1)", "(0,0,0)"),
            ("nan_origin", "(1,0,0) (0,1,0) (0,0,1)", "(0,nan,0)"),
        ]
    )
    @SkipIfNoModule("nrrd")
    def test_nrrd_not_finite(self, _, directions, origin):
        path = _write_nrrd(os.path.join(self.dir, "bad.nrrd"), directions, origin)
        reader = NrrdReader()
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            reader.get_data(reader.read(path))
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            LoadImage(image_only=True)(path)

    @SkipIfNoModule("nrrd")
    def test_nrrd_finite_loads(self):
        path = _write_nrrd(os.path.join(self.dir, "ok.nrrd"), "(2,0,0) (0,1,0) (0,0,1)", "(1,2,3)")
        img = LoadImage(image_only=True)(path)
        self.assertTrue(np.isfinite(np.asarray(img.affine)).all())
        self.assertEqual(tuple(img.shape), (4, 4, 4))

    @parameterized.expand(
        [("nan_origin", (np.nan, 0.0, 0.0), (1.0, 1.0, 1.0)), ("inf_spacing", (0.0, 0.0, 0.0), (np.inf, 1.0, 1.0))]
    )
    @SkipIfNoModule("itk")
    def test_itk_not_finite(self, _, origin, spacing):
        img = itk.image_from_array(np.ones((4, 4, 4), np.float32))
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            ITKReader().get_data(img)

    @SkipIfNoModule("itk")
    def test_itk_finite(self):
        img = itk.image_from_array(np.ones((4, 4, 4), np.float32))
        img.SetOrigin((1.0, 2.0, 3.0))
        img.SetSpacing((2.0, 1.0, 1.0))
        _, meta = ITKReader().get_data(img)
        self.assertTrue(np.isfinite(meta["affine"]).all())

    def test_load_image_checks_custom_reader(self):
        with self.assertRaisesRegex(ValueError, "_NanAffineReader returned a non-finite 'affine'"):
            LoadImage(image_only=True, reader=_NanAffineReader())("anything")

    def test_load_image_checks_runtime_reader(self):
        with self.assertRaisesRegex(ValueError, "non-finite 'affine'"):
            LoadImage(image_only=True)("anything", reader=_NanAffineReader())

    @SkipIfNoModule("itk")
    @SkipIfNoModule("pydicom")
    def test_itk_dicom_series_overflowing_positions(self):
        # GDCM clamps an out-of-range DS value to +/-DBL_MAX, and the inter-slice direction is
        # computed from the difference of two positions, so opposite-sign overflow makes it non-finite.
        pydicom, _ = optional_import("pydicom")
        pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE
        shape = (8, 8)
        for name, positions in (("bad", ["-1e400", "1e400"]), ("ok", ["0", "1"])):
            series_dir = os.path.join(self.dir, name)
            os.makedirs(series_dir)
            series_uid = pydicom.uid.generate_uid()
            for i, z in enumerate(positions):
                meta = pydicom.dataset.FileMetaDataset()
                meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
                meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                ds = pydicom.dataset.Dataset()
                ds.file_meta = meta
                ds.SOPClassUID = meta.MediaStorageSOPClassUID
                ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                ds.SeriesInstanceUID = series_uid
                ds.StudyInstanceUID = pydicom.uid.generate_uid()
                ds.Modality = "CT"
                ds.Rows, ds.Columns = shape
                ds.BitsAllocated = ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.PixelSpacing = ["1", "1"]
                ds.SliceThickness = "1"
                ds.InstanceNumber = i + 1
                ds.ImagePositionPatient = [z, "0", "0"]
                ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
                ds.PixelData = np.ones(shape, np.uint16).tobytes()
                ds.save_as(os.path.join(series_dir, f"slice_{i}.dcm"), enforce_file_format=True)

        reader = ITKReader()
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            reader.get_data(reader.read(os.path.join(self.dir, "bad")))
        with self.assertRaisesRegex(ValueError, NOT_FINITE):
            LoadImage(image_only=True)(os.path.join(self.dir, "bad"))
        # the control series still loads, so the check is not rejecting series in general
        img = LoadImage(image_only=True)(os.path.join(self.dir, "ok"))
        self.assertTrue(np.isfinite(np.asarray(img.affine)).all())
        self.assertEqual(tuple(img.shape), (8, 8, 2))


if __name__ == "__main__":
    unittest.main()
