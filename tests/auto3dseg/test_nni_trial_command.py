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
import subprocess
import tempfile
import unittest

from monai.apps.auto3dseg.auto_runner import _nni_trial_command

PREFIX = ["python", "-m", "monai.apps.auto3dseg", "NNIGen", "run_algo"]


@unittest.skipIf(os.name == "nt", "POSIX shell quoting")
class TestNNITrialCommand(unittest.TestCase):
    def test_plain_paths(self):
        cmd = _nni_trial_command("/work/algo_0/algo_object.json", "/work")
        self.assertEqual(cmd, "python -m monai.apps.auto3dseg NNIGen run_algo /work/algo_0/algo_object.json /work")

    def test_paths_with_special_characters_are_single_arguments(self):
        for work_dir in ("/tmp/my work", "/tmp/a;b", "/tmp/$(echo x)", "/tmp/`x`", "/tmp/a'b\"c", "/tmp/a|b&c"):
            with self.subTest(work_dir=work_dir):
                cmd = _nni_trial_command(os.path.join(work_dir, "algo_object.json"), work_dir)
                self.assertEqual(shlex.split(cmd), PREFIX + [os.path.join(work_dir, "algo_object.json"), work_dir])

    def test_shell_receives_the_literal_path(self):
        """Run the command through ``sh`` with ``python`` swapped for ``printf`` and check the arguments it sees."""
        with tempfile.TemporaryDirectory() as tempdir:
            marker = os.path.join(tempdir, "marker")
            work_dir = f"{tempdir}/w; touch {marker}; $(touch {marker})"
            cmd = _nni_trial_command("obj.json", work_dir)
            self.assertTrue(cmd.startswith("python "))
            out = subprocess.run(
                ["sh", "-c", "printf '%s\\n' " + cmd[len("python ") :]], capture_output=True, text=True, check=True
            ).stdout
            self.assertEqual(out.splitlines(), PREFIX[1:] + ["obj.json", work_dir])
            self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()
