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
import warnings
from unittest.mock import patch

from monai.config import vendor_deps
from monai.config.vendor_deps import REGISTRY, Vendor, active_vendor, canonical_name, register, split_requirement


class FakeVendor(Vendor):
    """A vendor whose every hook is set from the constructor."""

    name = "fake"

    def __init__(self, substitutions=None, extras=None, requirements=(), detected=False, hardware=False):
        self.substitutions = {} if substitutions is None else substitutions
        self.extras = extras or {}
        self.requirements = list(requirements)
        self.detected = detected
        self.hardware = hardware

    def detect(self):
        return self.detected

    def hardware_present(self):
        return self.hardware

    def extras_for(self, name):
        return self.extras.get(name, [])

    def extra_requirements(self):
        return list(self.requirements)


def only(*vendors, env=None):
    """Run ``active_vendor`` against exactly ``vendors``, importing no real plugin."""
    return (
        patch.dict("os.environ", env or {}, clear=True),
        patch.object(vendor_deps, "PLUGIN_MODULES", ()),
        patch.dict(REGISTRY, {vendor.name: vendor for vendor in vendors}, clear=True),
    )


class TestRequirementParsing(unittest.TestCase):
    def test_canonical_name(self):
        self.assertEqual("nvidia-ml-py", canonical_name("NVIDIA_ML.Py"))

    def test_split_plain(self):
        self.assertEqual(("numpy", [], ">=1.24,<3.0"), split_requirement("numpy>=1.24,<3.0"))

    def test_split_extras_and_marker(self):
        name, extras, rest = split_requirement("cupy-cuda13x[ctk]!=14.1.0; platform_system == 'Linux'")
        self.assertEqual("cupy-cuda13x", name)
        self.assertEqual(["ctk"], extras)
        self.assertEqual("!=14.1.0; platform_system == 'Linux'", rest)

    def test_split_direct_url(self):
        name, extras, _ = split_requirement("MetricsReloaded @ git+https://github.com/x/y@z")
        self.assertEqual("MetricsReloaded", name)
        self.assertEqual([], extras)


class TestRewrites(unittest.TestCase):
    def test_substitution_drop_and_dedupe(self):
        vendor = FakeVendor({"a-cu12": "repl", "a-cu13": "repl", "gone": None})
        self.assertEqual(
            {"g": ["repl", "kept"]}, vendor.apply_to_optional_dependencies({"g": ["a-cu12", "a-cu13", "gone", "kept"]})
        )

    def test_names_are_matched_canonically(self):
        self.assertEqual(["repl"], FakeVendor({"a-cu12": "repl"}).substitute(["A_cu12>=1.0"]))

    def test_group_can_become_empty(self):
        self.assertEqual({"g": []}, FakeVendor({"gone": None}).apply_to_optional_dependencies({"g": ["gone"]}))

    def test_extras_and_extra_requirements(self):
        vendor = FakeVendor(extras={"torch": ["device-x"]}, requirements=["runtime>=1"])
        self.assertEqual(
            ["torch[device-x]>=2.8.0", "numpy>=1.24", "runtime>=1"],
            vendor.apply_to_dependencies(["torch>=2.8.0", "numpy>=1.24"]),
        )

    def test_existing_extras_preserved(self):
        vendor = FakeVendor(extras={"torch": ["device-x"]})
        self.assertEqual(
            ["torch[device-x,opt-einsum]>=2.8.0"], vendor.apply_to_dependencies(["torch[opt-einsum]>=2.8.0"])
        )

    def test_unrelated_requirements_untouched(self):
        group = ["scikit-image>=0.19.0", "MetricsReloaded @ git+https://github.com/x/y@z"]
        self.assertEqual({"all": group}, FakeVendor().apply_to_optional_dependencies({"all": group}))

    def test_base_vendor_changes_nothing(self):
        dependencies = ["torch>=2.8.0", "cucim-cu13"]
        self.assertEqual(dependencies, Vendor().apply_to_dependencies(dependencies))


class TestRegister(unittest.TestCase):
    """A malformed substitution table would match nothing and leave the lists unmodified, silently."""

    def setUp(self):
        registry = patch.dict(REGISTRY, {}, clear=True)
        registry.start()
        self.addCleanup(registry.stop)

    def test_adds_to_registry(self):
        vendor = register(FakeVendor({"cucim-cu12": "amd-hipcim"}))
        self.assertIs(vendor, REGISTRY["fake"])

    def test_name_is_required(self):
        with self.assertRaises(ValueError):
            register(Vendor())

    def test_substitutions_must_be_a_mapping(self):
        for table in ("not a dict", ["cucim-cu12"], 7):
            with self.subTest(table=table), self.assertRaises(TypeError):
                register(FakeVendor(table))
        self.assertEqual({}, REGISTRY)

    def test_substitution_keys_must_be_canonical(self):
        # Lookups canonicalise the requirement name, so "cucim_cu12" could never match "cucim-cu12".
        for key in ("cucim_cu12", "cucim.cu12", "CuCIM-cu12", 7):
            with self.subTest(key=key), self.assertRaises(ValueError):
                register(FakeVendor({key: "amd-hipcim"}))
        self.assertEqual({}, REGISTRY)


class TestVendorSelection(unittest.TestCase):
    def test_none_disables_without_importing_plugins(self):
        for value in ("none", "NONE", "cpu"):
            with (
                patch.dict("os.environ", {"MONAI_VENDOR": value}),
                patch.object(vendor_deps.importlib, "import_module", side_effect=AssertionError("must not import")),
            ):
                self.assertIsNone(active_vendor())

    def test_plugins_register_themselves_on_import(self):
        with patch.dict("os.environ", {"MONAI_VENDOR": "rocm"}):
            vendor = active_vendor()
        self.assertEqual("rocm", vendor.name)
        self.assertIs(vendor, REGISTRY["rocm"])

    def test_unknown_vendor_is_rejected(self):
        env, plugins, registry = only(FakeVendor(), env={"MONAI_VENDOR": "nosuchvendor"})
        with env, plugins, registry, self.assertRaises(ValueError):
            active_vendor()

    def test_explicit_vendor_bypasses_detection(self):
        fake = FakeVendor(detected=False)
        env, plugins, registry = only(fake, env={"MONAI_VENDOR": "fake"})
        with env, plugins, registry:
            self.assertIs(fake, active_vendor())

    def test_detection_selects_matching_vendor(self):
        fake = FakeVendor(detected=True)
        env, plugins, registry = only(fake)
        with env, plugins, registry:
            self.assertIs(fake, active_vendor())

    def test_no_vendor_when_none_detects(self):
        env, plugins, registry = only(FakeVendor(detected=False))
        with env, plugins, registry:
            self.assertIsNone(active_vendor())

    def test_plugin_that_cannot_be_imported_is_fatal(self):
        # Carrying on without it would ship another vendor's packages in this vendor's wheel.
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vendor_deps, "PLUGIN_MODULES", ("vendor_does_not_exist",)),
            self.assertRaises(ImportError),
        ):
            active_vendor()


class TestHardwareWithoutVendor(unittest.TestCase):
    """``pip`` build isolation can hand the build a generic torch while sitting on vendor hardware."""

    def _warnings(self, vendor):
        env, plugins, registry = only(vendor)
        with env, plugins, registry, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = active_vendor()
        return selected, [str(w.message) for w in caught]

    def test_warns_when_hardware_present_but_torch_is_not(self):
        selected, messages = self._warnings(FakeVendor(detected=False, hardware=True))
        self.assertIsNone(selected)
        self.assertEqual(1, len(messages))
        self.assertIn("fake hardware is present", messages[0])
        self.assertIn("MONAI_VENDOR=fake", messages[0])

    def test_silent_when_hardware_absent(self):
        self.assertEqual((None, []), self._warnings(FakeVendor(detected=False, hardware=False)))

    def test_silent_when_the_vendor_is_active(self):
        fake = FakeVendor(detected=True, hardware=True)
        self.assertEqual((fake, []), self._warnings(fake))


if __name__ == "__main__":
    unittest.main()
