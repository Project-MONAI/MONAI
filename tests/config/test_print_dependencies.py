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
import re
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from parameterized import parameterized

from monai.config.print_dependencies import parse_dependencies, print_dependencies_argv

TEST_TOML = """
[build-system]
requires = ["setuptools", "wheel"]

[project]
name = "test"
dependencies = ["torch", "numpy"]

[project.optional-dependencies]
all = ["something", "another"]
testing = ["coverage", "black"]
recursive= ["test[testing]"]
recursive_weekly= ["test-weekly[testing]"]
"""

PARSE_CASES = [
    ([], ["numpy", "torch"]),
    (["testing"], ["black", "coverage", "numpy", "torch"]),
    (["build-system"], ["numpy", "setuptools", "torch", "wheel"]),
    (["*"], ["another", "black", "coverage", "numpy", "something", "torch"]),
    (["recursive"], ["black", "coverage", "numpy", "torch"]),
    (["recursive_weekly"], ["black", "coverage", "numpy", "torch"]),
]

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"
NVIDIA_ONLY = re.compile(r"cucim-cu|cupy-cuda|nvidia|nni\b", re.IGNORECASE)


@patch("monai.config.print_dependencies.PROJ", "test")
@patch("monai.config.print_dependencies.PROJ_WEEKLY", "test-weekly")
class TestPrintDependencies(unittest.TestCase):
    def setUp(self):
        self.toml = NamedTemporaryFile("w", delete=False)
        self.toml.write(TEST_TOML)
        self.toml.close()

    def tearDown(self):
        os.unlink(self.toml.name)

    @parameterized.expand(PARSE_CASES)
    def test_parse_dependencies(self, sections, outputs):
        deps = parse_dependencies(self.toml.name, sections)
        self.assertEqual(outputs, deps)

    def test_missing_section(self):
        with self.assertRaises(KeyError):
            parse_dependencies(self.toml.name, ["nonexistent_section"])

    def test_print_dependencies(self):
        out = StringIO()
        with redirect_stdout(out), patch("monai.config.print_dependencies.TOML_FILE", self.toml.name):

            with self.subTest("Test correct print"):
                with patch("sys.argv", ["", "all", "build-system", "*"]):
                    print_dependencies_argv()

                self.assertGreater(out.tell(), 0)

            with self.subTest("Test missing section"):
                with patch("sys.argv", ["", "nonexistent_section"]), self.assertRaises(KeyError):
                    print_dependencies_argv()


@unittest.skipUnless(PYPROJECT.is_file(), "requires pyproject.toml from the source tree")
class TestAmdExtras(unittest.TestCase):
    def test_all_amd_has_no_nvidia_packages(self):
        """`all_amd` is for ROCm machines, where packages needing NVIDIA hardware install but do not work."""
        deps = parse_dependencies(str(PYPROJECT), ["all_amd"])
        self.assertEqual([], [d for d in deps if NVIDIA_ONLY.match(d)])
        for package in ("amd-cupy", "amd-hipcim"):
            self.assertTrue(any(d.startswith(package) for d in deps), f"{package} missing from all_amd")


if __name__ == "__main__":
    unittest.main()
