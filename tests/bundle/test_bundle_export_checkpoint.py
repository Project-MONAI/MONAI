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

import json
import os
import tempfile
import unittest
from pathlib import Path

from parameterized import parameterized

from monai.bundle import ConfigParser
from monai.data import load_exported_program
from monai.networks import save_state
from tests.test_utils import command_line_tests, skip_if_windows

TESTS_PATH = Path(__file__).parents[1]

# key_in_ckpt
TEST_CASE_1 = [""]
TEST_CASE_2 = ["model"]


@skip_if_windows
class TestExportCheckpoint(unittest.TestCase):
    def setUp(self):
        self._orig_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")

    def tearDown(self):
        if self._orig_cuda_env is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._orig_cuda_env
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_export(self, key_in_ckpt):
        meta_file = os.path.join(TESTS_PATH, "testing_data", "metadata.json")
        config_file = os.path.join(TESTS_PATH, "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg"}
            def_args_file = os.path.join(tempdir, "def_args.yaml")

            ckpt_file = os.path.join(tempdir, "model.pt")
            pt2_file = os.path.join(tempdir, "model.pt2")

            parser = ConfigParser()
            parser.export_config_file(config=def_args, filepath=def_args_file)
            parser.read_config(config_file)
            net = parser.get_parsed_content("network_def")
            save_state(src=net if key_in_ckpt == "" else {key_in_ckpt: net}, path=ckpt_file)

            cmd = [
                "coverage",
                "run",
                "-m",
                "monai.bundle",
                "export_checkpoint",
                "network_def",
                "--filepath",
                pt2_file,
                "--meta_file",
                meta_file,
                "--config_file",
                f"['{config_file}','{def_args_file}']",
                "--ckpt_file",
                ckpt_file,
                "--key_in_ckpt",
                key_in_ckpt,
                "--args_file",
                def_args_file,
                "--input_shape",
                "[1, 1, 96, 96, 96]",
            ]
            command_line_tests(cmd)
            self.assertTrue(os.path.exists(pt2_file))

            _, _metadata, extra_files = load_exported_program(
                pt2_file, more_extra_files=["inference.json", "def_args.json"]
            )
            self.assertIn("meta_file", json.loads(extra_files["def_args.json"]))
            self.assertIn("network_def", json.loads(extra_files["inference.json"]))

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_default_value(self, key_in_ckpt):
        config_file = os.path.join(TESTS_PATH, "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            def_args = {"meta_file": "will be replaced by `meta_file` arg"}
            def_args_file = os.path.join(tempdir, "def_args.yaml")
            ckpt_file = os.path.join(tempdir, "models", "model.pt")
            pt2_file = os.path.join(tempdir, "models", "model.pt2")

            parser = ConfigParser()
            parser.export_config_file(config=def_args, filepath=def_args_file)
            parser.read_config(config_file)
            net = parser.get_parsed_content("network_def")
            save_state(src=net if key_in_ckpt == "" else {key_in_ckpt: net}, path=ckpt_file)

            # check with default value
            cmd = [
                "coverage",
                "run",
                "-m",
                "monai.bundle",
                "export_checkpoint",
                "--key_in_ckpt",
                key_in_ckpt,
                "--config_file",
                config_file,
                "--bundle_root",
                tempdir,
                "--input_shape",
                "[1, 1, 96, 96, 96]",
            ]
            command_line_tests(cmd)
            self.assertTrue(os.path.exists(pt2_file))


if __name__ == "__main__":
    unittest.main()
