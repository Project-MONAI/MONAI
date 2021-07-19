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

import torch
from parameterized import parameterized

from monai.transforms import SaveImaged
from monai.utils.module import optional_import
from tests.utils import TEST_NDARRAYS

_, has_pil = optional_import("PIL")
_, has_nib = optional_import("nibabel")

exts = [ext for has_lib, ext in zip((has_nib, has_pil), (".nii.gz", ".png")) if has_lib]

TESTS = []
for p in TEST_NDARRAYS:
    for ext in exts:
        TESTS.append(
            [
                {
                    "img": p(torch.randint(0, 255, (1, 2, 3, 4))),
                    "img_meta_dict": {"filename_or_obj": "testfile0" + ext},
                },
                ext,
                False,
            ]
        )
        TESTS.append(
            [
                {
                    "img": p(torch.randint(0, 255, (1, 2, 3, 4))),
                    "img_meta_dict": {"filename_or_obj": "testfile0" + ext},
                    "patch_index": 6,
                },
                ext,
                False,
            ]
        )


class TestSaveImaged(unittest.TestCase):
    @parameterized.expand(TESTS, skip_on_empty=True)
    def test_saved_content(self, test_data, output_ext, resample):
        with tempfile.TemporaryDirectory() as tempdir:
            trans = SaveImaged(
                keys=["img", "pred"],
                meta_keys="img_meta_dict",
                output_dir=tempdir,
                output_ext=output_ext,
                resample=resample,
                allow_missing_keys=True,
            )
            trans(test_data)

            patch_index = test_data["img_meta_dict"].get("patch_index", None)
            patch_index = f"_{patch_index}" if patch_index is not None else ""
            filepath = os.path.join("testfile0", "testfile0" + "_trans" + patch_index + output_ext)
            self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))


if __name__ == "__main__":
    unittest.main()
