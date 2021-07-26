# Copyright 2020 - 2021 MONAI Consortium
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
from parameterized import parameterized

import numpy as np
import torch

from monai.data import ITKReader, ITKWriter

TEST_CASE_1 = [".nii.gz", np.float32, None, None, None, False, torch.zeros(8, 1, 2, 3, 4), (4, 3, 2)]

TEST_CASE_2 = [
    ".dcm",
    np.uint8,
    [np.diag(np.ones(4)) * 1.0 for _ in range(8)],
    [np.diag(np.ones(4)) * 5.0 for _ in range(8)],
    [(10, 10, 2) for _ in range(8)],
    True,
    torch.zeros(8, 3, 2, 3, 4),
    (10, 10, 2, 3),
]


class TestITKWriter(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_saved_content(self, output_ext, output_dtype, affine, original_affine, output_shape, resample, data, expected_shape):
        with tempfile.TemporaryDirectory() as tempdir:

            writer = ITKWriter(
                output_dir=tempdir,
                output_postfix="seg",
                output_ext=output_ext,
                output_dtype=output_dtype,
                resample=resample,
            )

            meta_data = {"filename_or_obj": ["testfile" + str(i) + ".nii.gz" for i in range(8)]}
            if output_shape is not None:
                meta_data["spatial_shape"] = output_shape
            if affine is not None:
                meta_data["affine"] = affine
            if original_affine is not None:
                meta_data["original_affine"] = original_affine

            for i in range(8):
                writer.write(data=data[i], meta_data={k: meta_data[k][i] for k in meta_data} if meta_data is not None else None)
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg" + output_ext)
                reader = ITKReader()
                img = reader.read(data=os.path.join(tempdir, filepath))
                result, meta = reader.get_data(img)
                self.assertTupleEqual(result.shape, expected_shape)
                if affine is not None:
                    # no need to compare the last line of affine matrix
                    np.testing.assert_allclose(meta["affine"][:-1], original_affine[i][:-1])


if __name__ == "__main__":
    unittest.main()
