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
lint = ["ruff", "black"]
testing = ["test[lint]", "coverage"]
cyclic = ["test[cyclic]", "spam"]
spell_check = ["eggs"]
mixed = ["Test[Spell.Check]", "ham"]
other_pkg = ["versioneer[toml]", "bacon"]
"""

# a project whose name needs normalizing before a self-reference spelled differently can match it
PUNCTUATED_NAME_TOML = """
[project]
name = "My_Project"
dependencies = ["torch"]

[project.optional-dependencies]
lint = ["ruff"]
testing = ["my-project[lint]", "coverage"]
"""

PARSE_CASES = [
    ([], ["numpy", "torch"]),
    # "test[lint]" expands rather than reaching pip as a requirement on the published package
    (["testing"], ["black", "coverage", "numpy", "ruff", "torch"]),
    (["build-system"], ["numpy", "setuptools", "torch", "wheel"]),
    (
        ["*"],
        [
            "another",
            "bacon",
            "black",
            "coverage",
            "eggs",
            "ham",
            "numpy",
            "ruff",
            "something",
            "spam",
            "torch",
            "versioneer[toml]",
        ],
    ),
    (["cyclic"], ["numpy", "spam", "torch"]),
    # PEP 685: extra names compare equal across case and "-"/"_"/"." spellings
    (["mixed"], ["eggs", "ham", "numpy", "torch"]),
    # another project's extra is not a self-reference and passes through untouched
    (["other_pkg"], ["bacon", "numpy", "torch", "versioneer[toml]"]),
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

    def test_self_reference_spelled_differently(self):
        """PEP 503: "my-project[lint]" is a self-reference to a project named "My_Project"."""
        toml = NamedTemporaryFile("w", delete=False)
        toml.write(PUNCTUATED_NAME_TOML)
        toml.close()
        self.addCleanup(os.unlink, toml.name)

        self.assertEqual(["coverage", "ruff", "torch"], parse_dependencies(toml.name, ["testing"]))

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
