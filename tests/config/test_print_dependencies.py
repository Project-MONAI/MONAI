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

import unittest
from contextlib import redirect_stdout
from io import StringIO
from tempfile import NamedTemporaryFile
from unittest.mock import patch

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
"""

TOML_FILE = "monai.config.print_dependencies.TOML_FILE"


class TestPrintDependencies(unittest.TestCase):

    def test_parse_dependencies(self):
        with NamedTemporaryFile("w") as fp:
            fp.write(TEST_TOML)
            fp.flush()

            with self.subTest("Test default sections"):
                deps = parse_dependencies(fp.name)
                self.assertEqual(["numpy", "torch"], deps)

            with self.subTest("Test some sections"):
                deps = parse_dependencies(fp.name, ["testing"])
                self.assertEqual(["black", "coverage", "numpy", "torch"], deps)

            with self.subTest("Test build-system sections"):
                deps = parse_dependencies(fp.name, ["build-system"])
                self.assertEqual(["numpy", "setuptools", "torch", "wheel"], deps)

            with self.subTest("Test all sections"):
                deps = parse_dependencies(fp.name, ["*"])
                self.assertEqual(["another", "black", "coverage", "numpy", "something", "torch"], deps)

            with self.subTest("Test missing section"):
                with self.assertRaises(KeyError):
                    parse_dependencies(fp.name, ["nonexistent_section"])

    def test_print_dependencies(self):
        out = StringIO()
        with NamedTemporaryFile("w") as fp, redirect_stdout(out), patch(TOML_FILE, fp.name):
            fp.write(TEST_TOML)
            fp.flush()

            with self.subTest("Test correct print"):
                with patch("sys.argv", ["", "all", "build-system", "*"]):
                    print_dependencies_argv()

                self.assertGreater(out.tell(), 0)

            with self.subTest("Test missing section"):
                with patch("sys.argv", ["", "nonexistent_section"]), self.assertRaises(KeyError):
                    print_dependencies_argv()


if __name__ == "__main__":
    unittest.main()
