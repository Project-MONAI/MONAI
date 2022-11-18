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
import os
import tempfile
import unittest

from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.data import load_net_with_metadata
from monai.networks import save_state
from tests.utils import command_line_tests, skip_if_windows

TEST_CASE_1 = [""]

TEST_CASE_2 = ["model"]


@skip_if_windows
class TestCKPTExport(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_export(self, key_in_ckpt):
        meta_file = os.path.join(os.path.dirname(__file__), "testing_data", "metadata.json")
        config_file = os.path.join(os.path.dirname(__file__), "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg"}
            def_args_file = os.path.join(tempdir, "def_args.yaml")

            ckpt_file = os.path.join(tempdir, "model.pt")
            ts_file = os.path.join(tempdir, "model.ts")

            parser = ConfigParser()
            parser.export_config_file(config=def_args, filepath=def_args_file)
            parser.read_config(config_file)
            net = parser.get_parsed_content("network_def")
            save_state(src=net if key_in_ckpt == "" else {key_in_ckpt: net}, path=ckpt_file)

            cmd = ["coverage", "run", "-m", "monai.bundle", "ckpt_export", "network_def", "--filepath", ts_file]
            cmd += ["--meta_file", meta_file, "--config_file", f"['{config_file}','{def_args_file}']", "--ckpt_file"]
            cmd += [ckpt_file, "--key_in_ckpt", key_in_ckpt, "--args_file", def_args_file]
            command_line_tests(cmd)
            self.assertTrue(os.path.exists(ts_file))

            _, metadata, extra_files = load_net_with_metadata(
                ts_file, more_extra_files=["inference.json", "def_args.json"]
            )
            self.assertTrue("schema" in metadata)
            self.assertTrue("meta_file" in json.loads(extra_files["def_args.json"]))
            self.assertTrue("network_def" in json.loads(extra_files["inference.json"]))


if __name__ == "__main__":
    unittest.main()
