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
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from monai.apps.auto3dseg.hpo_gen import NNIGen, _nni_trial_command, _quote_or_raw, _quote_trial_arg
from tests.test_utils import skip_if_windows

PREFIX = ["python", "-m", "monai.apps.auto3dseg", "NNIGen", "run_algo"]
SPECIAL_POSIX = ("my work", "a;b", "$(echo x)", "`x`", "a'b\"c", "a|b&c", "a\u2019b", "%PATH%", "a!b")
SPECIAL_WINDOWS = ("my work", "a;b", "$(echo x)", "`x`", "a'b", "a|b&c", "a\u2019b", "a^b", "(x)", "a<b>c")
ECHO_ARGS = "import json, sys; print(json.dumps(sys.argv[1:]))"


def _run_through_shell(cmd: str, tempdir: str) -> list[str]:
    """
    Run ``cmd`` with the platform shell, as NNI does, with ``python`` swapped for a script printing its arguments.

    ``subprocess.run(..., shell=True)`` uses ``/bin/sh`` on POSIX and ``cmd.exe`` on Windows, matching the shells
    NNI's local training service runs ``trialCommand`` with.

    Args:
        cmd: a command produced by ``_nni_trial_command``.
        tempdir: a directory for the helper script.

    Returns:
        the arguments the program received.
    """
    script = os.path.join(tempdir, "echo_args.py")
    with open(script, "w") as f:
        f.write(ECHO_ARGS)
    if sys.platform == "win32":
        program = subprocess.list2cmdline([sys.executable, script])
    else:
        program = shlex.join([sys.executable, script])
    full_cmd = program + cmd[len("python") :]
    out = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, check=True, cwd=tempdir).stdout
    return json.loads(out)


class TestNNITrialCommandPosix(unittest.TestCase):
    def test_plain_paths(self):
        cmd = _nni_trial_command("/work/algo_0/algo_object.json", "/work", windows=False)
        self.assertEqual(cmd, "python -m monai.apps.auto3dseg NNIGen run_algo /work/algo_0/algo_object.json /work")

    def test_paths_with_special_characters_are_single_arguments(self):
        for name in SPECIAL_POSIX:
            with self.subTest(name=name):
                work_dir = f"/tmp/{name}"
                cmd = _nni_trial_command(f"{work_dir}/algo_object.json", work_dir, windows=False)
                self.assertEqual(shlex.split(cmd), PREFIX + [f"{work_dir}/algo_object.json", work_dir])

    @skip_if_windows
    def test_shell_receives_the_literal_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            marker = os.path.join(tempdir, "marker")
            for name in SPECIAL_POSIX + (f"w; touch {marker}", f"$(touch {marker})"):
                with self.subTest(name=name):
                    work_dir = os.path.join(tempdir, name)
                    args = _run_through_shell(_nni_trial_command("obj.json", work_dir), tempdir)
                    self.assertEqual(args, PREFIX[1:] + ["obj.json", work_dir])
                    self.assertFalse(os.path.exists(marker))


class TestNNITrialCommandWindows(unittest.TestCase):
    def test_cmd_quoting(self):
        self.assertEqual(_quote_trial_arg("C:\\work dir", windows=True), '"C:\\work dir"')
        self.assertEqual(_quote_trial_arg("C:\\", windows=True), '"C:\\\\"')
        self.assertEqual(
            _nni_trial_command("C:\\w\\obj.json", "C:\\w", windows=True),
            'python -m monai.apps.auto3dseg NNIGen run_algo "C:\\w\\obj.json" "C:\\w"',
        )

    def test_characters_cmd_cannot_quote_are_rejected(self):
        for name in ('a"b', "%PATH%", "a!b", "a\nb", "a\x00b"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    _nni_trial_command("obj.json", f"C:\\{name}", windows=True)

    @unittest.skipUnless(sys.platform == "win32", "Requires Windows cmd.exe")
    def test_cmd_receives_the_literal_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            marker = os.path.join(tempdir, "marker")
            for name in SPECIAL_WINDOWS + (f"w & type nul > {marker}", f"w | type nul > {marker}"):
                with self.subTest(name=name):
                    work_dir = os.path.join(tempdir, name)
                    args = _run_through_shell(_nni_trial_command("obj.json", work_dir), tempdir)
                    self.assertEqual(args, PREFIX[1:] + ["obj.json", work_dir])
                    self.assertFalse(os.path.exists(marker))


class TestPrintedInstruction(unittest.TestCase):
    def setUp(self):
        self.gen = NNIGen()
        self.gen.obj_filename = "C:\\data\\100%\\algo_object.json"
        self.gen.algo = cast(Any, SimpleNamespace(template_path="/templates", get_output_path=lambda: "/work/algo_0"))

    def test_unquotable_path_still_prints_the_instructions(self):
        with mock.patch("monai.apps.auto3dseg.hpo_gen.logger") as logger, mock.patch("sys.platform", "win32"):
            self.gen.print_bundle_algo_instruction()

        printed = "\n".join(str(call.args[0]) for call in logger.info.call_args_list)
        self.assertIn("C:\\data\\100%\\algo_object.json {result_dir}", printed)
        self.assertIn("{remote_algo_dir} {result_dir} {remote_algorithm_templates_dir}", printed)
        self.assertEqual(logger.warning.call_count, 1)

    def test_quotable_path_is_still_quoted(self):
        with mock.patch("monai.apps.auto3dseg.hpo_gen.logger") as logger, mock.patch("sys.platform", "win32"):
            self.gen.obj_filename = "C:\\data\\my algo\\algo_object.json"
            self.gen.print_bundle_algo_instruction()

        printed = "\n".join(str(call.args[0]) for call in logger.info.call_args_list)
        self.assertIn('"C:\\data\\my algo\\algo_object.json" {result_dir}', printed)
        logger.warning.assert_not_called()


class TestQuoteOrRaw(unittest.TestCase):
    def test_falls_back_to_the_raw_text(self):
        with mock.patch("monai.apps.auto3dseg.hpo_gen.logger"), mock.patch("sys.platform", "win32"):
            self.assertEqual(_quote_or_raw("C:\\100%\\obj.json"), "C:\\100%\\obj.json")

    def test_quotes_what_the_shell_accepts(self):
        self.assertEqual(_quote_or_raw("/tmp/my work/obj.json", windows=False), "'/tmp/my work/obj.json'")


if __name__ == "__main__":
    unittest.main()
