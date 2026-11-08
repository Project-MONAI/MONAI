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

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from parameterized import parameterized

from monai.bundle import ConfigParser, export_checkpoint, load
from monai.data import load_exported_program, save_exported_program
from monai.networks import save_state
from tests.test_utils import command_line_tests, skip_if_no_cuda, skip_if_windows

TESTS_PATH = Path(__file__).parents[1]

# key_in_ckpt
TEST_CASE_1 = [""]
TEST_CASE_2 = ["model"]


@skip_if_windows
class TestExportCheckpoint(unittest.TestCase):
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

            _loaded, _metadata, extra_files = load_exported_program(
                pt2_file, more_extra_files=["inference.json", "def_args.json"]
            )
            self.assertIn("meta_file", json.loads(extra_files["def_args.json"]))
            self.assertIn("network_def", json.loads(extra_files["inference.json"]))

            exported_weights = _loaded.state_dict
            for name, value in net.state_dict().items():
                torch.testing.assert_close(exported_weights[name], value)

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

    def test_missing_file_in_meta_list(self):
        config_file = os.path.join(TESTS_PATH, "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            parser = ConfigParser()
            parser.read_config(config_file)
            ckpt_file = os.path.join(tempdir, "model.pt")
            save_state(src=parser.get_parsed_content("network_def"), path=ckpt_file)
            meta_file = os.path.join(tempdir, "first.json")
            parser.export_config_file(config={"first": 1}, filepath=meta_file)
            with self.assertRaises(FileNotFoundError):
                export_checkpoint(
                    filepath=os.path.join(tempdir, "model.pt2"),
                    ckpt_file=ckpt_file,
                    meta_file=[meta_file, os.path.join(tempdir, "missing.json")],
                    config_file=config_file,
                    input_shape=[1, 1, 96, 96, 96],
                )

    @skip_if_no_cuda
    def test_network_on_gpu_exports_on_cpu(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_file = os.path.join(tempdir, "inference.json")
            net_def = {"_target_": "UNet", "spatial_dims": 2, "in_channels": 1, "out_channels": 2}
            net_def.update({"channels": [4, 8], "strides": [2]})
            config = {"network_def": net_def, "network": "$@network_def.to('cuda:0')"}
            ConfigParser.export_config_file(config=config, filepath=config_file)
            ckpt_file = os.path.join(tempdir, "model.pt")
            parser = ConfigParser()
            parser.read_config(config_file)
            save_state(src=parser.get_parsed_content("network_def"), path=ckpt_file)
            pt2_file = os.path.join(tempdir, "model.pt2")

            export_checkpoint(
                "network", filepath=pt2_file, ckpt_file=ckpt_file, config_file=config_file, input_shape=[1, 1, 16, 16]
            )
            module = load_exported_program(pt2_file)[0].module()

        self.assertEqual(next(module.parameters()).device.type, "cpu")
        self.assertEqual(module(torch.rand(1, 1, 16, 16)).shape, torch.Size([1, 2, 16, 16]))

    @skip_if_no_cuda
    def test_gpu_checkpoint_exports_without_gpu(self):
        """A checkpoint saved from a GPU can be exported where no GPU is visible."""
        with tempfile.TemporaryDirectory() as tempdir:
            config_file = os.path.join(tempdir, "inference.json")
            net_def = {"_target_": "UNet", "spatial_dims": 2, "in_channels": 1, "out_channels": 2}
            net_def.update({"channels": [4, 8], "strides": [2]})
            ConfigParser.export_config_file(config={"network_def": net_def}, filepath=config_file)
            ckpt_file = os.path.join(tempdir, "model.pt")
            parser = ConfigParser()
            parser.read_config(config_file)
            save_state(src=parser.get_parsed_content("network_def").to("cuda:0"), path=ckpt_file)
            pt2_file = os.path.join(tempdir, "model.pt2")

            cmd = ["coverage", "run", "-m", "monai.bundle", "export_checkpoint", "network_def"]
            cmd += ["--filepath", pt2_file, "--ckpt_file", ckpt_file, "--config_file", config_file]
            cmd += ["--input_shape", "[1, 1, 16, 16]"]
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                command_line_tests(cmd)
            self.assertTrue(os.path.exists(pt2_file))

    def test_meta_files_are_merged(self):
        config_file = os.path.join(TESTS_PATH, "testing_data", "inference.json")
        with tempfile.TemporaryDirectory() as tempdir:
            parser = ConfigParser()
            parser.read_config(config_file)
            ckpt_file = os.path.join(tempdir, "model.pt")
            save_state(src=parser.get_parsed_content("network_def"), path=ckpt_file)
            meta_files = [os.path.join(tempdir, "first.json"), os.path.join(tempdir, "second.json")]
            for meta_file, value in zip(meta_files, ({"first": 1}, {"second": 2})):
                parser.export_config_file(config=value, filepath=meta_file)
            pt2_file = os.path.join(tempdir, "model.pt2")

            export_checkpoint(
                filepath=pt2_file,
                ckpt_file=ckpt_file,
                meta_file=meta_files,
                config_file=config_file,
                input_shape=[1, 1, 96, 96, 96],
            )
            _, metadata, _ = load_exported_program(pt2_file)

        self.assertEqual(metadata, {"first": 1, "second": 2})


class TestLoadExportedModule(unittest.TestCase):
    def setUp(self):
        self.net = torch.nn.Linear(2, 2).eval()
        self.data = torch.rand(1, 2)
        self.exported = torch.export.export(self.net, args=(self.data,))

    def test_load_default_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.makedirs(os.path.join(tempdir, "bundle", "models"))
            save_exported_program(
                self.exported,
                os.path.join(tempdir, "bundle", "models", "model.pt2"),
                include_config_vals=False,
                meta_values={"version": "1"},
                more_extra_files={"inference.json": "{}"},
            )
            loaded, meta, extra = load(
                "bundle", bundle_dir=tempdir, load_exported_module=True, config_files=("inference.json",)
            )

        torch.testing.assert_close(loaded.module()(self.data), self.net(self.data))
        self.assertEqual(meta, {"version": "1"})
        self.assertEqual(extra, {"inference.json": "{}"})

    def test_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            load("bundle", load_ts_module=True, load_exported_module=True)

    def test_new_argument_is_last(self):
        self.assertEqual(list(inspect.signature(load).parameters)[-1], "load_exported_module")


if __name__ == "__main__":
    unittest.main()
