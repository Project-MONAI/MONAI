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
import unittest
from contextlib import redirect_stdout
from io import StringIO
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
"""

PARSE_CASES = [
    ([], ["numpy", "torch"]),
    (["testing"], ["black", "coverage", "numpy", "torch"]),
    (["build-system"], ["numpy", "setuptools", "torch", "wheel"]),
    (["*"], ["another", "black", "coverage", "numpy", "something", "torch"]),
]


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


if __name__ == "__main__":
    unittest.main()
