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
from urllib.error import ContentTooShortError, HTTPError

from monai.apps import download_and_extract, download_url, extractall
from tests.utils import skip_if_quick


class TestDownloadAndExtract(unittest.TestCase):
    @skip_if_quick
    def test_actions(self):
        testing_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        url = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
        filepath = os.path.join(testing_dir, "MedNIST.tar.gz")
        output_dir = testing_dir
        md5_value = "0bc7306e7427e00ad1c5526a6677552d"
        try:
            download_and_extract(url, filepath, output_dir, md5_value)
            download_and_extract(url, filepath, output_dir, md5_value)
        except (ContentTooShortError, HTTPError, RuntimeError) as e:
            print(str(e))
            if isinstance(e, RuntimeError):
                # FIXME: skip MD5 check as current downloading method may fail
                self.assertTrue(str(e).startswith("md5 check"))
            return  # skipping this test due the network connection errors

        wrong_md5 = "0"
        try:
            download_url(url, filepath, wrong_md5)
        except (ContentTooShortError, HTTPError, RuntimeError) as e:
            print(str(e))
            if isinstance(e, RuntimeError):
                # FIXME: skip MD5 check as current downloading method may fail
                self.assertTrue(str(e).startswith("md5 check"))
            return  # skipping this test due the network connection errors

        try:
            extractall(filepath, output_dir, wrong_md5)
        except RuntimeError as e:
            self.assertTrue(str(e).startswith("md5 check"))

    @skip_if_quick
    def test_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # icon.tar.gz https://drive.google.com/file/d/1aHztVYbynCCq-byTf6cRI8ViETpBWHyC/view?usp=sharing
                download_and_extract(
                    "https://drive.google.com/uc?id=1aHztVYbynCCq-byTf6cRI8ViETpBWHyC",
                    output_dir=tmp_dir,
                    hash_val="a55d11ad26ed9eb7277905d796205531",
                    file_type="tar",
                )
                # favicon.ico.zip https://drive.google.com/file/d/1YGa8N3LBddOkxBnnLndGicoW0klsAH6D/view?usp=sharing
                download_and_extract(
                    "https://drive.google.com/uc?id=1YGa8N3LBddOkxBnnLndGicoW0klsAH6D",
                    output_dir=tmp_dir,
                    hash_val="ac6e167ee40803577d98237f2b0241e5",
                    file_type="zip",
                )
            except (ContentTooShortError, HTTPError, RuntimeError) as e:
                print(str(e))
                if isinstance(e, RuntimeError):
                    # FIXME: skip MD5 check as current downloading method may fail
                    self.assertTrue(str(e).startswith("md5 check"))
                return  # skipping this test due the network connection errors


if __name__ == "__main__":
    unittest.main()
