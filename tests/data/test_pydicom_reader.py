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

from monai.data import PydicomReader
from monai.utils import MetaKeys
from tests.test_utils import SkipIfNoModule


@SkipIfNoModule("pydicom")
class TestPydicomReaderAffine(unittest.TestCase):
    def test_missing_orientation_tags_warns_and_returns_identity(self):
        # Without ImageOrientationPatient (0020,0037) and ImagePositionPatient
        # (0020,0032) the affine cannot be derived. The reader falls back to the
        # identity matrix; regression test for #8468 ensures this is no longer
        # silent so users know orientation/spacing may be wrong.
        reader = PydicomReader()
        with self.assertWarns(UserWarning):
            affine = reader._get_affine({})
        np.testing.assert_array_equal(affine, np.eye(4))

    def test_partial_orientation_tags_warns(self):
        # Only one of the two required tags present is still insufficient.
        reader = PydicomReader()
        metadata = {"00200037": {"Value": [1, 0, 0, 0, 1, 0]}}  # orientation only
        with self.assertWarns(UserWarning):
            affine = reader._get_affine(metadata)
        np.testing.assert_array_equal(affine, np.eye(4))

    def test_non_finite_pixel_spacing_raises(self):
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200032": {"Value": [0.0, 0.0, 0.0]},
            "00280030": {"Value": [np.nan, 1.0]},
        }
        with self.assertRaisesRegex(ValueError, "PixelSpacing"):
            reader._get_affine(metadata, lps_to_ras=False)

    def test_non_finite_image_position_raises(self):
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200032": {"Value": [np.inf, 0.0, 0.0]},
            "00280030": {"Value": [1.0, 1.0]},
        }
        with self.assertRaisesRegex(ValueError, "ImagePositionPatient"):
            reader._get_affine(metadata, lps_to_ras=False)

    def test_finite_values_return_affine(self):
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200032": {"Value": [10.0, 20.0, 30.0]},
            "00280030": {"Value": [0.5, 0.25]},
        }
        affine = reader._get_affine(metadata, lps_to_ras=False)
        self.assertEqual(affine.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(affine)))
        np.testing.assert_allclose(affine[0, 3], 10.0)
        np.testing.assert_allclose(affine[1, 3], 20.0)
        np.testing.assert_allclose(affine[2, 3], 30.0)

    def test_non_finite_orientation_raises(self):
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [np.nan, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200032": {"Value": [0.0, 0.0, 0.0]},
            "00280030": {"Value": [1.0, 1.0]},
        }
        with self.assertRaisesRegex(ValueError, "ImageOrientationPatient"):
            reader._get_affine(metadata, lps_to_ras=False)

    def test_non_finite_last_image_position_raises(self):
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
            "00200032": {"Value": [0.0, 0.0, 0.0]},
            "00280030": {"Value": [1.0, 1.0]},
            "lastImagePositionPatient": [0.0, 0.0, np.inf],
            MetaKeys.SPATIAL_SHAPE: [1, 1, 2],
        }
        with self.assertRaisesRegex(ValueError, "lastImagePositionPatient"):
            reader._get_affine(metadata, lps_to_ras=False)

    def test_overflow_from_finite_inputs_raises(self):
        # Finite inputs whose product overflows produce a non-finite affine.
        reader = PydicomReader()
        metadata = {
            "00200037": {"Value": [1e308, 0.0, 0.0, 1e308, 0.0, 0.0]},
            "00200032": {"Value": [0.0, 0.0, 0.0]},
            "00280030": {"Value": [1e308, 1e308]},
        }
        with self.assertRaisesRegex(ValueError, "not finite"):
            reader._get_affine(metadata, lps_to_ras=False)



if __name__ == "__main__":
    unittest.main()
