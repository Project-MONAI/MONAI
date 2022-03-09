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

import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest

from parameterized import parameterized

TEST_CASE_1 = [os.path.join(os.path.dirname(__file__), "testing_data", "metadata.json")]

url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/" "meta_schema_202202281232.json"


class TestVerifyMeta(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_verify(self, metafile):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "schema.json")
            resultfile = os.path.join(tempdir, "results.txt")
            hash_val = "486c581cca90293d1a7f41a57991ff35"

            cmd = (
                f"{sys.executable} -m monai.bundle.verify_meta -m {metafile} -u {url} -f {filepath}"
                f" -r {resultfile} -v {hash_val}"
            )
            ret = subprocess.check_call(cmd.split(" "))
            self.assertEqual(ret, 0)
            self.assertFalse(os.path.exists(resultfile))

    def test_verify_error(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, "schema.json")
            metafile = os.path.join(tempdir, "metadata.json")
            with open(metafile, "w") as f:
                json.dump({"wrong_meta": "wrong content"}, f)
            resultfile = os.path.join(tempdir, "results.txt")

            cmd = f"{sys.executable} -m monai.bundle.verify_meta -m {metafile} -u {url} -f {filepath} -r {resultfile}"
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.check_call(cmd.split(" "))
            self.assertTrue(os.path.exists(resultfile))


if __name__ == "__main__":
    unittest.main()
