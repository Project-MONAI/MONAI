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

import torch
from parameterized import parameterized

import monai.networks.nets as nets
from monai.apps import check_hash
from monai.bundle import ConfigParser, load
from tests.utils import (
    SkipIfBeforePyTorchVersion,
    command_line_tests,
    skip_if_downloading_fails,
    skip_if_quick,
    skip_if_windows,
)

TEST_CASE_1 = ["test_bundle", None]

TEST_CASE_2 = ["test_bundle_v0.1.1", None]

TEST_CASE_3 = ["test_bundle", "0.1.1"]

TEST_CASE_4 = [
    ["model.pt", "model.ts", "network.json", "test_output.pt", "test_input.pt"],
    "test_bundle",
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/test_bundle.zip",
    "a131d39a0af717af32d19e565b434928",
]

TEST_CASE_5 = [
    ["model.pt", "model.ts", "network.json", "test_output.pt", "test_input.pt"],
    "test_bundle",
    "Project-MONAI/MONAI-extra-test-data/0.8.1",
    "cuda" if torch.cuda.is_available() else "cpu",
    "model.pt",
]

TEST_CASE_6 = [
    ["test_output.pt", "test_input.pt"],
    "test_bundle",
    "0.1.1",
    "Project-MONAI/MONAI-extra-test-data/0.8.1",
    "cuda" if torch.cuda.is_available() else "cpu",
    "model.ts",
]


@skip_if_windows
class TestDownload(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    @skip_if_quick
    def test_download_bundle(self, bundle_name, version):
        bundle_files = ["model.pt", "model.ts", "network.json", "test_output.pt", "test_input.pt"]
        repo = "Project-MONAI/MONAI-extra-test-data/0.8.1"
        hash_val = "a131d39a0af717af32d19e565b434928"
        with skip_if_downloading_fails():
            # download a whole bundle from github releases
            with tempfile.TemporaryDirectory() as tempdir:
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--name", bundle_name, "--source", "github"]
                cmd += ["--bundle_dir", tempdir, "--repo", repo, "--progress", "False"]
                if version is not None:
                    cmd += ["--version", version]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, "test_bundle", file)
                    self.assertTrue(os.path.exists(file_path))
                    if file == "network.json":
                        self.assertTrue(check_hash(filepath=file_path, val=hash_val))

    @parameterized.expand([TEST_CASE_4])
    @skip_if_quick
    def test_url_download_bundle(self, bundle_files, bundle_name, url, hash_val):
        with skip_if_downloading_fails():
            # download a single file from url, also use `args_file`
            with tempfile.TemporaryDirectory() as tempdir:
                def_args = {"name": bundle_name, "bundle_dir": tempdir, "url": ""}
                def_args_file = os.path.join(tempdir, "def_args.json")
                parser = ConfigParser()
                parser.export_config_file(config=def_args, filepath=def_args_file)
                cmd = ["coverage", "run", "-m", "monai.bundle", "download", "--args_file", def_args_file]
                cmd += ["--url", url]
                command_line_tests(cmd)
                for file in bundle_files:
                    file_path = os.path.join(tempdir, bundle_name, file)
                    self.assertTrue(os.path.exists(file_path))
                if file == "network.json":
                    self.assertTrue(check_hash(filepath=file_path, val=hash_val))


class TestLoad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_5])
    @skip_if_quick
    def test_load_weights(self, bundle_files, bundle_name, repo, device, model_file):
        with skip_if_downloading_fails():
            # download bundle, and load weights from the downloaded path
            with tempfile.TemporaryDirectory() as tempdir:
                # load weights
                weights = load(
                    name=bundle_name,
                    model_file=model_file,
                    bundle_dir=tempdir,
                    repo=repo,
                    progress=False,
                    device=device,
                )

                # prepare network
                with open(os.path.join(tempdir, bundle_name, bundle_files[2])) as f:
                    net_args = json.load(f)["network_def"]
                model_name = net_args["_target_"]
                del net_args["_target_"]
                model = nets.__dict__[model_name](**net_args)
                model.to(device)
                model.load_state_dict(weights)
                model.eval()

                # prepare data and test
                input_tensor = torch.load(os.path.join(tempdir, bundle_name, bundle_files[4]), map_location=device)
                output = model.forward(input_tensor)
                expected_output = torch.load(os.path.join(tempdir, bundle_name, bundle_files[3]), map_location=device)
                torch.testing.assert_allclose(output, expected_output)

                # load instantiated model directly and test, since the bundle has been downloaded,
                # there is no need to input `repo`
                model_2 = load(
                    name=bundle_name,
                    model_file=model_file,
                    bundle_dir=tempdir,
                    progress=False,
                    device=device,
                    net_name=model_name,
                    **net_args,
                )
                model_2.eval()
                output_2 = model_2.forward(input_tensor)
                torch.testing.assert_allclose(output_2, expected_output)

    @parameterized.expand([TEST_CASE_6])
    @skip_if_quick
    @SkipIfBeforePyTorchVersion((1, 7, 1))
    def test_load_ts_module(self, bundle_files, bundle_name, version, repo, device, model_file):
        with skip_if_downloading_fails():
            # load ts module
            with tempfile.TemporaryDirectory() as tempdir:
                # load ts module
                model_ts, metadata, extra_file_dict = load(
                    name=bundle_name,
                    version=version,
                    model_file=model_file,
                    load_ts_module=True,
                    bundle_dir=tempdir,
                    repo=repo,
                    progress=False,
                    device=device,
                    config_files=("network.json",),
                )

                # prepare and test ts
                input_tensor = torch.load(os.path.join(tempdir, bundle_name, bundle_files[1]), map_location=device)
                output = model_ts.forward(input_tensor)
                expected_output = torch.load(os.path.join(tempdir, bundle_name, bundle_files[0]), map_location=device)
                torch.testing.assert_allclose(output, expected_output)
                # test metadata
                self.assertTrue(metadata["pytorch_version"] == "1.7.1")
                # test extra_file_dict
                self.assertTrue("network.json" in extra_file_dict.keys())


if __name__ == "__main__":
    unittest.main()
