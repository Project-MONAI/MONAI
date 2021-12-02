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

import os
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data import ITKWriter
from monai.transforms import LoadImage
from monai.utils import optional_import
from tests.utils import TEST_NDARRAYS, assert_allclose

itk, has_itk = optional_import("itk")
nib, has_nibabel = optional_import("nibabel")

TEST_CASES_AFFINE = []
for p in TEST_NDARRAYS:
    case_1d = p([[1.0, 0.0], [1.0, 1.0]]), p([[-1.0, 0.0], [1.0, 1.0]])
    TEST_CASES_AFFINE.append(case_1d)
    case_2d_1 = p([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]), p([[-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
    TEST_CASES_AFFINE.append(case_2d_1)
    case_2d_2 = p([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), p(
        [[-1.0, 0.0, -1.0], [0.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    )
    TEST_CASES_AFFINE.append(case_2d_2)
    case_3d = p([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 3.0]]), p(
        [[-1.0, 0.0, -1.0, -1.0], [0.0, -1.0, -1.0, -2.0], [1.0, 1.0, 1.0, 3.0]]
    )
    TEST_CASES_AFFINE.append(case_3d)
    case_4d = p(np.ones((5, 5))), p([[-1] * 5, [-1] * 5, [1] * 5, [1] * 5, [1] * 5])
    TEST_CASES_AFFINE.append(case_4d)


@unittest.skipUnless(has_itk, "Requires `itk` package.")
@unittest.skipUnless(has_nibabel, "Requires `nibabel` package.")
class TestITKWriter(unittest.TestCase):
    def test_channel_shape(self):
        with tempfile.TemporaryDirectory() as tempdir:
            for c in (0, 1, 2, 3):
                fname = os.path.join(tempdir, f"testing{c}.nii")
                (
                    ITKWriter()
                    .set_data_array(torch.zeros(1, 2, 3, 4), channel_dim=c, squeeze_end_dims=False)
                    .set_metadata({})
                    .write(fname)
                )
                itk_obj = itk.imread(fname)
                s = [1, 2, 3, 4]
                s.pop(c)
                np.testing.assert_allclose(itk.size(itk_obj), s)

    @parameterized.expand(TEST_CASES_AFFINE)
    def test_ras_to_lps(self, param, expected):
        assert_allclose(ITKWriter.ras_to_lps(param), expected)

    def test_nifti_writing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            testing_file = os.path.join(os.path.dirname(__file__), "testing_data", "anatomical.nii")
            writer_file = os.path.join(tempdir, "testing_itk_anatomical.nii")
            loader_dtype, writer_dtype = np.float32, np.float32

            data_array_ref, metadata_ref = LoadImage(reader="nibabelreader", dtype=loader_dtype)(testing_file)
            writer = ITKWriter(output_dtype=writer_dtype)
            writer.set_data_array(data_array_ref, channel_dim=None).set_metadata(metadata_ref)
            writer.write(writer_file)
            data_array, metadata = LoadImage(reader="nibabelreader", dtype=loader_dtype)(writer_file)
            self.assertEqual(data_array.shape, data_array_ref.shape)
            for expected, actual in zip(metadata_ref, metadata):
                self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
