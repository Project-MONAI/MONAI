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

import glob
import os
import unittest

import monai


class TestAllImport(unittest.TestCase):
    def test_public_api(self):
        """
        This is to check "monai.__all__" should be consistent with
        the top-level folders except for "__pycache__", "_extensions" and "csrc" (cpp/cuda src)
        """
        base_folder = os.path.dirname(monai.__file__)
        to_search = os.path.join(base_folder, "*", "")
        subfolders = [os.path.basename(x[:-1]) for x in glob.glob(to_search)]
        to_exclude = ("__pycache__", "_extensions", "csrc")
        mod = []
        for code_folder in subfolders:
            if code_folder in to_exclude:
                continue
            mod.append(code_folder)
        self.assertEqual(sorted(monai.__all__), sorted(mod))


if __name__ == "__main__":
    unittest.main()
