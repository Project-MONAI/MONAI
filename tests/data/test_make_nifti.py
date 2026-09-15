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

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data.synthetic import create_test_image_2d
from monai.utils import optional_import
from tests.test_utils import make_nifti_image

_, has_nib = optional_import("nibabel")

TESTS = []
caller_owned_dir = tempfile.mkdtemp()
atexit.register(shutil.rmtree, caller_owned_dir, ignore_errors=True)
for affine in (None, np.eye(4), torch.eye(4)):
    for dir in (None, caller_owned_dir):
        for fname in (None, "fname"):
            TESTS.append([{"affine": affine, "dir": dir, "fname": fname}])


@unittest.skipUnless(has_nib, "Requires nibabel")
class TestMakeNifti(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_make_nifti(self, params):
        im, _ = create_test_image_2d(100, 88)
        created_file = make_nifti_image(im, verbose=True, **params)
        self.assertTrue(os.path.isfile(created_file))

    def test_temp_dir_removed_at_exit(self):
        script = (
            "from monai.data.synthetic import create_test_image_2d;"
            "from tests.test_utils import make_nifti_image;"
            "print(make_nifti_image(create_test_image_2d(100, 88)[0]))"
        )
        created_file = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        self.assertFalse(os.path.exists(os.path.dirname(created_file)))

    def test_caller_owned_dir_kept(self):
        im, _ = create_test_image_2d(100, 88)
        with tempfile.TemporaryDirectory() as caller_dir:
            make_nifti_image(im, dir=caller_dir)
            self.assertTrue(os.path.isdir(caller_dir))


if __name__ == "__main__":
    unittest.main()
