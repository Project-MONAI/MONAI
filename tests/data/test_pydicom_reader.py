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


if __name__ == "__main__":
    unittest.main()
