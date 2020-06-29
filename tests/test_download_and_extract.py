# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import shutil
import numpy as np
import tempfile
from monai.application import download_and_extract, download_url, extractall
from tests.utils import skip_if_quick


class TestDownloadAndExtract(unittest.TestCase):
    @skip_if_quick
    def test_actions(self):
        tempdir = tempfile.mkdtemp()
        url = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
        filepath = os.path.join(tempdir, "MedNIST.tar.gz")
        output_dir = tempdir
        md5_value = "0bc7306e7427e00ad1c5526a6677552d"
        download_and_extract(url, filepath, output_dir, md5_value)
        download_and_extract(url, filepath, output_dir, md5_value)

        wrong_md5 = "0"
        try:
            download_url(url, filepath, wrong_md5)
        except RuntimeError as e:
            self.assertTrue(str(e).startswith("MD5 check"))

        shutil.rmtree(os.path.join(tempdir, "MedNIST"))
        try:
            extractall(filepath, output_dir, wrong_md5)
        except RuntimeError as e:
            self.assertTrue(str(e).startswith("MD5 check"))

        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
