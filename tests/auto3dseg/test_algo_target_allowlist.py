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

from monai.auto3dseg.utils import _reject_non_algo_target, algo_from_json


class TestAlgoTargetAllowlist(unittest.TestCase):
    """Regression tests for GHSA-2wx3-8x3w-r8qv.

    ``algo_from_json`` resolves the file's ``_target_`` through ``ConfigParser``, which imports the
    dotted path and calls it. ``algo_object.json`` has exactly one legitimate target type, so a
    ``_target_`` that is not an ``Algo`` subclass is rejected before it is instantiated.
    """

    def test_code_execution_targets_are_rejected(self):
        for target in ("subprocess.call", "os.system", "builtins.eval", "builtins.exec", "shutil.rmtree"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError) as ctx:
                    _reject_non_algo_target(target, "algo_object.json")
                self.assertIn("GHSA-2wx3-8x3w-r8qv", str(ctx.exception))

    def test_non_class_target_is_rejected(self):
        """A module or plain function is not an Algo subclass."""
        with self.assertRaisesRegex(ValueError, r"GHSA-2wx3-8x3w-r8qv"):
            _reject_non_algo_target("json.dumps", "algo_object.json")

    def test_unresolvable_target_raises_module_not_found(self):
        """Unresolvable names raise ModuleNotFoundError so the caller can try the next path."""
        with self.assertRaises(ModuleNotFoundError):
            _reject_non_algo_target("no_such_module.NoSuchAlgo", "algo_object.json")

    def test_algo_subclass_is_accepted(self):
        """A real Algo subclass passes the check."""
        _reject_non_algo_target("monai.apps.auto3dseg.BundleAlgo", "algo_object.json")

    def test_algo_from_json_rejects_payload_target(self):
        """End to end: a malicious algo_object.json never reaches instantiation."""
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "algo_object.json")
            marker = os.path.join(tempdir, "PWNED")
            with open(path, "w") as f:
                json.dump({"_target_": "subprocess.call", "args": ["/bin/sh", "-c", f"touch {marker}"]}, f)

            with self.assertRaisesRegex(ValueError, r"GHSA-2wx3-8x3w-r8qv"):
                algo_from_json(path)
            self.assertFalse(os.path.exists(marker), "the algo_object.json payload executed")


if __name__ == "__main__":
    unittest.main()
