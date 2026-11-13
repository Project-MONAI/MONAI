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
import sys
import tempfile
import textwrap
import unittest
import uuid

from monai.apps.auto3dseg.bundle_gen import BundleGen

_ALGO_TEMPLATE = """
from monai.auto3dseg.algo_gen import Algo


class LiteralAlgo(Algo):
    def __init__(self, template_path=None, **kwargs):
        self.template_path = template_path
        self.extra = kwargs
"""


class TestBundleGenAlgosLiterals(unittest.TestCase):
    """Regression tests for the ``algos`` mapping of ``BundleGen``.

    The mapping is user input: ``AutoRunner`` reads it from ``input.yaml`` and ``BundleGen``/its CLI accept it
    directly. Its values are plain data, so they must reach the ``Algo`` constructor verbatim instead of being
    parsed by ``ConfigParser``, which evaluates strings starting with ``$``, ``@`` or ``%``.
    """

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.root = cls.tempdir.name
        # A uniquely named package so repeated runs and other test modules never share a module object.
        cls.package = f"literal_algos_{uuid.uuid4().hex}"
        package_dir = os.path.join(cls.root, cls.package)
        os.makedirs(os.path.join(package_dir, "scripts"))
        for init in (os.path.join(package_dir, "__init__.py"), os.path.join(package_dir, "scripts", "__init__.py")):
            with open(init, "w") as f:
                f.write("")
        with open(os.path.join(package_dir, "scripts", "algo.py"), "w") as f:
            f.write(textwrap.dedent(_ALGO_TEMPLATE))
        cls.target = f"{cls.package}.scripts.algo.LiteralAlgo"
        cls.marker = os.path.join(cls.root, "PWNED")

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == cls.package or name.startswith(f"{cls.package}."):
                del sys.modules[name]
        if cls.root in sys.path:
            sys.path.remove(cls.root)
        cls.tempdir.cleanup()

    def setUp(self):
        if os.path.exists(self.marker):
            os.unlink(self.marker)
        # Start from a clean import state so each test is independent of the ones before it.
        if self.root in sys.path:
            sys.path.remove(self.root)
        for name in list(sys.modules):
            if name == self.package or name.startswith(f"{self.package}."):
                del sys.modules[name]

    def _payload(self, value: str = "pwned") -> str:
        return f"$__import__('pathlib').Path({self.marker!r}).write_text({value!r})"

    def test_mapping_values_reach_the_constructor_literally(self):
        """A ``$`` value in the mapping is passed through as a string and never evaluated."""
        payload = self._payload()
        gen = BundleGen(algos={"algo1": {"_target_": self.target, "template_path": self.root, "note": payload}})

        self.assertFalse(os.path.exists(self.marker), "the algos mapping payload was evaluated")
        self.assertEqual(len(gen.algos), 1)
        algo = gen.algos[0]
        self.assertEqual(type(algo).__name__, "LiteralAlgo")
        self.assertEqual(algo.name, "algo1")
        self.assertEqual(algo.template_path, self.root)
        self.assertEqual(algo.extra["note"], payload)

    def test_nested_config_directive_is_kept_as_plain_data(self):
        """A nested ``_target_`` mapping is not resolved; it stays a dict and nothing is instantiated from it."""
        nested = {"_target_": "os.system", "args": [f"touch {self.marker}"]}
        gen = BundleGen(algos={"algo1": {"_target_": self.target, "template_path": self.root, "note": nested}})

        self.assertFalse(os.path.exists(self.marker), "a nested _target_ in the algos mapping was resolved")
        self.assertEqual(gen.algos[0].extra["note"], nested)

    def test_reference_and_file_strings_are_literal(self):
        """``@`` and ``%`` strings in the mapping reach the constructor as plain strings."""
        for value in ("@some_id", "%other.json#key"):
            with self.subTest(value=value):
                gen = BundleGen(algos={"algo1": {"_target_": self.target, "template_path": self.root, "note": value}})
                self.assertEqual(gen.algos[0].extra["note"], value)

    def test_short_target_name_is_resolved(self):
        """A bare monai component name, as bundle syntax allows, still instantiates the class."""
        gen = BundleGen(algos={"algo1": {"_target_": "BundleAlgo", "template_path": self.root}})

        self.assertEqual(len(gen.algos), 1)
        self.assertEqual(type(gen.algos[0]).__name__, "BundleAlgo")
        self.assertEqual(gen.algos[0].template_path, self.root)

    def test_expression_as_template_path_is_not_evaluated(self):
        """A ``$`` expression used as ``template_path`` is not evaluated, even though the module then fails."""
        payload = self._payload()
        with self.assertRaises(ModuleNotFoundError):
            BundleGen(algos={"algo1": {"_target_": self.target, "template_path": payload}})
        self.assertFalse(os.path.exists(self.marker), "the template_path payload was evaluated")

    def test_non_algo_target_is_rejected(self):
        """``_target_`` must name an ``Algo`` subclass, not an arbitrary callable."""
        with self.assertRaisesRegex(ValueError, r"GHSA-2wx3-8x3w-r8qv"):
            BundleGen(algos={"algo1": {"_target_": "os.system", "args": [f"touch {self.marker}"]}})
        self.assertFalse(os.path.exists(self.marker), "a non-Algo _target_ was instantiated")

    def test_string_entry_raises_value_error(self):
        """The documented mapping shape is a dict per algorithm; a bare string is rejected cleanly."""
        with self.assertRaisesRegex(ValueError, r"expected a dict"):
            BundleGen(algos={"algo1": self._payload()})
        self.assertFalse(os.path.exists(self.marker))

    def test_missing_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"missing '_target_' key"):
            BundleGen(algos={"algo1": {"template_path": self.root}})

    def test_non_string_template_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"expected a string"):
            BundleGen(algos={"algo1": {"_target_": self.target, "template_path": ["not", "a", "path"]}})


if __name__ == "__main__":
    unittest.main()
