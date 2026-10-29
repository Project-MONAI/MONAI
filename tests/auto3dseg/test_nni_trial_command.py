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
import shlex
import shutil
import subprocess
import tempfile
import unittest

from monai.apps.auto3dseg.hpo_gen import _nni_trial_command, _quote_trial_arg

PREFIX = ["python", "-m", "monai.apps.auto3dseg", "NNIGen", "run_algo"]
SPECIAL = ("my work", "a;b", "$(echo x)", "`x`", "a'b\"c", "a|b&c", "a\u2019b", "$env:PATH", "@(1)")
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


class TestNNITrialCommandPosix(unittest.TestCase):
    def test_plain_paths(self):
        cmd = _nni_trial_command("/work/algo_0/algo_object.json", "/work", windows=False)
        self.assertEqual(cmd, "python -m monai.apps.auto3dseg NNIGen run_algo /work/algo_0/algo_object.json /work")

    def test_paths_with_special_characters_are_single_arguments(self):
        for name in SPECIAL:
            with self.subTest(name=name):
                work_dir = f"/tmp/{name}"
                cmd = _nni_trial_command(f"{work_dir}/algo_object.json", work_dir, windows=False)
                self.assertEqual(shlex.split(cmd), PREFIX + [f"{work_dir}/algo_object.json", work_dir])

    @unittest.skipIf(os.name == "nt" or not shutil.which("sh"), "Requires a POSIX sh")
    def test_shell_receives_the_literal_path(self):
        """Run the command through ``sh`` with ``python`` swapped for ``printf`` and check the arguments it sees."""
        with tempfile.TemporaryDirectory() as tempdir:
            marker = os.path.join(tempdir, "marker")
            work_dir = f"{tempdir}/w; touch {marker}; $(touch {marker})"
            cmd = _nni_trial_command("obj.json", work_dir, windows=False)
            out = subprocess.run(
                ["sh", "-c", "printf '%s\\n' " + cmd[len("python ") :]], capture_output=True, text=True, check=True
            ).stdout
            self.assertEqual(out.splitlines(), PREFIX[1:] + ["obj.json", work_dir])
            self.assertFalse(os.path.exists(marker))


class TestNNITrialCommandPowerShell(unittest.TestCase):
    def test_powershell_literal_quoting(self):
        self.assertEqual(_quote_trial_arg("C:\\work dir", windows=True), "'C:\\work dir'")
        self.assertEqual(_quote_trial_arg("C:\\O'Neil", windows=True), "'C:\\O''Neil'")
        self.assertEqual(_quote_trial_arg("a\u2019b", windows=True), "'a\u2019\u2019b'")
        self.assertEqual(
            _nni_trial_command("C:\\w\\obj.json", "C:\\w", windows=True),
            "python -m monai.apps.auto3dseg NNIGen run_algo 'C:\\w\\obj.json' 'C:\\w'",
        )

    @unittest.skipUnless(POWERSHELL, "Requires PowerShell")
    def test_powershell_receives_the_literal_path(self):
        """Run the command through PowerShell with ``python`` swapped for a function that echoes its arguments."""
        with tempfile.TemporaryDirectory() as tempdir:
            marker = os.path.join(tempdir, "marker")
            for name in SPECIAL + (f"w; New-Item {marker}", f"$(New-Item {marker})"):
                with self.subTest(name=name):
                    work_dir = os.path.join(tempdir, name)
                    cmd = _nni_trial_command("obj.json", work_dir, windows=True)
                    script = "function python { $args | ForEach-Object { Write-Output $_ } }; " + cmd
                    out = subprocess.run(
                        [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-Command", script],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        check=True,
                    ).stdout
                    self.assertEqual(out.splitlines(), PREFIX[1:] + ["obj.json", work_dir])
                    self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()
