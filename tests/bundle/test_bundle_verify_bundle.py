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
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from monai.bundle import ConfigParser, verify_bundle
from tests.test_utils import command_line_tests, download_url_or_skip_test, skip_if_windows, testing_data_config

TESTS_DIR = Path(__file__).parents[1]
TESTING_DATA = TESTS_DIR / "testing_data"
SCHEMA_FILE = TESTING_DATA / "schema.json"


class TestVerifyBundle(unittest.TestCase):
    def setUp(self):
        self.config = testing_data_config("configs", "test_meta_file")
        self.tempdir = tempfile.mkdtemp()
        self.bundle = Path(self.tempdir) / "test_bundle"
        (self.bundle / "configs").mkdir(parents=True)
        (self.bundle / "models").mkdir()
        shutil.copy(TESTING_DATA / "metadata.json", self.bundle / "configs" / "metadata.json")
        shutil.copy(TESTING_DATA / "inference.json", self.bundle / "configs" / "inference.json")
        (self.bundle / "LICENSE").write_text("test license")
        parser = ConfigParser()
        parser.read_config(str(self.bundle / "configs" / "inference.json"))
        torch.save(parser.get_parsed_content("network_def").state_dict(), self.bundle / "models" / "model.pt")

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)
        for name in [m for m in sys.modules if m == "scripts" or m.startswith("scripts.")]:
            sys.modules.pop(name)

    def _download_schema(self):
        download_url_or_skip_test(
            url=self.config["url"],
            filepath=SCHEMA_FILE,
            hash_val=self.config.get("hash_val"),
            hash_type=self.config.get("hash_type", "sha256"),
        )

    def _update_metadata(self, **kwargs):
        meta_file = self.bundle / "configs" / "metadata.json"
        metadata = json.loads(meta_file.read_text())
        metadata.update(kwargs)
        meta_file.write_text(json.dumps(metadata))

    def test_min(self):
        self._download_schema()
        verify_bundle(bundle_path=str(self.bundle), schema_file=str(SCHEMA_FILE))
        self.assertEqual(sorted(os.listdir(self.bundle)), ["LICENSE", "configs", "models"])  # bundle is unchanged

    def test_full(self):
        self._download_schema()
        verify_bundle(bundle_path=str(self.bundle), mode="full", schema_file=str(SCHEMA_FILE), device="cpu")
        self.assertEqual(os.listdir(self.bundle / "models"), ["model.pt"])  # the test export is not kept

    def test_full_with_custom_code(self):
        self._download_schema()
        # a bundle network defined in the bundle's own `scripts` package
        (self.bundle / "scripts").mkdir()
        (self.bundle / "scripts" / "__init__.py").write_text("")
        (self.bundle / "scripts" / "net.py").write_text(
            "from monai.networks.nets import UNet\n\n\nclass CustomUNet(UNet):\n    pass\n"
        )
        config_path = self.bundle / "configs" / "inference.json"
        config = json.loads(config_path.read_text())
        config["network_def"]["_target_"] = "scripts.net.CustomUNet"
        config_path.write_text(json.dumps(config))

        sys_path = list(sys.path)
        verify_bundle(bundle_path=str(self.bundle), mode="full", schema_file=str(SCHEMA_FILE), device="cpu")
        self.assertEqual(sys.path, sys_path)  # `sys.path` is restored

    def test_skip_net_io(self):
        self._download_schema()
        # the network outputs 2 channels, so verifying the input and output fails with this metadata
        meta_file = self.bundle / "configs" / "metadata.json"
        metadata = json.loads(meta_file.read_text())
        metadata["network_data_format"]["outputs"]["pred"]["num_channels"] = 3
        meta_file.write_text(json.dumps(metadata))
        kwargs = {"bundle_path": str(self.bundle), "mode": "full", "schema_file": str(SCHEMA_FILE), "device": "cpu"}
        with self.assertRaisesRegex(ValueError, "channel"):
            verify_bundle(**kwargs)
        verify_bundle(check_net_io=False, **kwargs)

    def test_skip_torchscript(self):
        self._download_schema()
        # without a checkpoint, the TorchScript export fails
        (self.bundle / "models" / "model.pt").unlink()
        kwargs = {"bundle_path": str(self.bundle), "mode": "full", "schema_file": str(SCHEMA_FILE), "device": "cpu"}
        with self.assertRaises(FileNotFoundError):
            verify_bundle(check_preferred_files=False, **kwargs)
        verify_bundle(check_preferred_files=False, check_torchscript=False, **kwargs)

    @skip_if_windows
    def test_cli(self):
        self._download_schema()
        cmd = ["coverage", "run", "-m", "monai.bundle", "verify_bundle", "--bundle_path", str(self.bundle)]
        cmd += ["--mode", "full", "--schema_file", str(SCHEMA_FILE), "--device", "cpu"]
        command_line_tests(cmd)

    def test_missing_license(self):
        (self.bundle / "LICENSE").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "LICENSE"):
            verify_bundle(bundle_path=str(self.bundle))

    def test_missing_checkpoint(self):
        (self.bundle / "models" / "model.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "model.pt"):
            verify_bundle(bundle_path=str(self.bundle))

    def test_missing_inference_config(self):
        (self.bundle / "configs" / "inference.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "inference"):
            verify_bundle(bundle_path=str(self.bundle))

    def test_skip_preferred_files(self):
        self._download_schema()
        (self.bundle / "models" / "model.pt").unlink()
        (self.bundle / "configs" / "inference.json").unlink()
        verify_bundle(bundle_path=str(self.bundle), schema_file=str(SCHEMA_FILE), check_preferred_files=False)

    def test_version_not_in_changelog(self):
        self._update_metadata(version="9.9.9")
        with self.assertRaisesRegex(ValueError, "9.9.9"):
            verify_bundle(bundle_path=str(self.bundle))

    def test_missing_changelog(self):
        meta_file = self.bundle / "configs" / "metadata.json"
        metadata = json.loads(meta_file.read_text())
        metadata.pop("changelog")
        meta_file.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "changelog"):
            verify_bundle(bundle_path=str(self.bundle))

    def test_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            verify_bundle(bundle_path=str(self.bundle), mode="regular")

    def test_missing_bundle_dir(self):
        with self.assertRaises(FileNotFoundError):
            verify_bundle(bundle_path=str(self.bundle / "not_exist"))


if __name__ == "__main__":
    unittest.main()
