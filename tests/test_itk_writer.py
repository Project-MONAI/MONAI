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
class TestITKWriter(unittest.TestCase):
    def test_channel_shape(self):
        with tempfile.TemporaryDirectory() as tempdir:
            for c in (0, 1, 2, 3):
                fname = os.path.join(tempdir, f"testing{c}.nii")
                itk_writer = ITKWriter()
                itk_writer.set_data_array(torch.zeros(1, 2, 3, 4), channel_dim=c, squeeze_end_dims=False)
                itk_writer.set_metadata({})
                itk_writer.write(fname)
                itk_obj = itk.imread(fname)
                s = [1, 2, 3, 4]
                s.pop(c)
                np.testing.assert_allclose(itk.size(itk_obj), s)

    @parameterized.expand(TEST_CASES_AFFINE)
    def test_ras_to_lps(self, param, expected):
        assert_allclose(ITKWriter.ras_to_lps(param), expected)


if __name__ == "__main__":
    unittest.main()
