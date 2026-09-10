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

import os
import tempfile
import unittest
from unittest import mock

from monai.apps.nnunet.nnunetv2_runner import nnUNetV2Runner
from monai.utils import OptionalImportError


class TestnnUNetV2RunnerOptionalImport(unittest.TestCase):
    """``nnUNetV2Runner`` requires the optional ``nnunetv2`` package to be installed."""

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        test_path = self.test_dir.name
        self.input_config = {
            "dataset_name_or_id": "123",
            "dataroot": os.path.join(test_path, "data"),
            "datalist": os.path.join(test_path, "lists", "task4.json"),
            "work_dir": os.path.join(test_path, "work"),
            "nnunet_raw": os.path.join(test_path, "nnUNet_raw"),
            "nnunet_preprocessed": os.path.join(test_path, "nnUNet_preprocessed"),
            "nnunet_results": os.path.join(test_path, "nnUNet_results"),
        }

    def test_missing_nnunetv2_raises_optional_import_error(self) -> None:
        """A missing ``nnunetv2`` must be reported as such, not as a bare ``ModuleNotFoundError``."""
        with mock.patch("monai.utils.module.optional_import", return_value=(None, False)):
            with self.assertRaises(OptionalImportError) as context:
                nnUNetV2Runner(input_config=dict(self.input_config))
        self.assertIn("nnunetv2", str(context.exception))

    def test_missing_nnunetv2_does_not_warn_about_the_dataset(self) -> None:
        """The dataset lookup warning must not fire when the real cause is the missing package."""
        with mock.patch("monai.utils.module.optional_import", return_value=(None, False)):
            with self.assertNoLogs("monai.apps.nnunet.nnunetv2_runner", level="WARNING"):
                with self.assertRaises(OptionalImportError):
                    nnUNetV2Runner(input_config=dict(self.input_config))

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
