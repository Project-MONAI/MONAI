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

import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from monai.config import vendor_deps
from monai.config.vendor_deps import (
    active_vendor,
    apply_to_dependencies,
    apply_to_optional_dependencies,
    canonical_name,
    split_requirement,
)


def fake_plugin(substitutions=None, extras=None, extra_requirements=()):
    """A minimal plugin satisfying the contract documented in vendor_deps."""
    return types.SimpleNamespace(
        NAME="fake",
        SUBSTITUTIONS=substitutions or {},
        extras_for=lambda name: (extras or {}).get(name, []),
        extra_requirements=lambda: list(extra_requirements),
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


class TestGenericRewrites(unittest.TestCase):
    def test_substitution_drop_and_dedupe(self):
        plugin = fake_plugin({"a-cu12": "repl", "a-cu13": "repl", "gone": None})
        self.assertEqual(
            {"g": ["repl", "kept"]}, apply_to_optional_dependencies({"g": ["a-cu12", "a-cu13", "gone", "kept"]}, plugin)
        )

    def test_group_can_become_empty(self):
        self.assertEqual({"g": []}, apply_to_optional_dependencies({"g": ["gone"]}, fake_plugin({"gone": None})))

    def test_extras_and_extra_requirements(self):
        plugin = fake_plugin(extras={"torch": ["device-x"]}, extra_requirements=["runtime>=1"])
        self.assertEqual(
            ["torch[device-x]>=2.8.0", "numpy>=1.24", "runtime>=1"],
            apply_to_dependencies(["torch>=2.8.0", "numpy>=1.24"], plugin),
        )

    def test_existing_extras_preserved(self):
        plugin = fake_plugin(extras={"torch": ["device-x"]})
        self.assertEqual(
            ["torch[device-x,opt-einsum]>=2.8.0"], apply_to_dependencies(["torch[opt-einsum]>=2.8.0"], plugin)
        )

    def test_unrelated_requirements_untouched(self):
        group = ["scikit-image>=0.19.0", "MetricsReloaded @ git+https://github.com/x/y@z"]
        self.assertEqual({"all": group}, apply_to_optional_dependencies({"all": group}, fake_plugin()))


class TestVendorSelection(unittest.TestCase):
    def test_none_disables_rewriting(self):
        for value in ("none", "NONE", "cpu"):
            with patch.dict("os.environ", {"MONAI_VENDOR": value}):
                self.assertIsNone(active_vendor())

    def test_unknown_vendor_is_rejected(self):
        with patch.dict("os.environ", {"MONAI_VENDOR": "nosuchvendor"}), self.assertRaises(ValueError):
            active_vendor()

    def test_explicit_vendor_bypasses_detection(self):
        with patch.dict("os.environ", {"MONAI_VENDOR": "rocm"}):
            vendor = active_vendor()
        self.assertIsNotNone(vendor)
        self.assertEqual("rocm", vendor.name)

    def test_no_vendor_when_no_probe_matches(self):
        with patch.dict("os.environ", {}, clear=True), patch.dict(vendor_deps.REGISTRY, {}, clear=True):
            self.assertIsNone(active_vendor())

    def test_plugin_not_imported_when_probe_does_not_match(self):
        probe = vendor_deps._Registration("vendor_rocm", lambda: False)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True),
            patch.object(vendor_deps, "_import_plugin", side_effect=AssertionError("plugin must not be imported")),
        ):
            self.assertIsNone(active_vendor())

    def test_detection_selects_matching_plugin(self):
        probe = vendor_deps._Registration("vendor_rocm", lambda: True)
        with patch.dict("os.environ", {}, clear=True), patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True):
            vendor = active_vendor()
        self.assertEqual("rocm", vendor.name)

    def test_detected_vendor_that_cannot_load_is_fatal(self):
        # Falling back to "no vendor" here would ship another vendor's packages in this one's wheel.
        probe = vendor_deps._Registration("vendor_broken", lambda: True)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True),
            patch.object(vendor_deps, "_import_plugin", side_effect=SyntaxError("invalid syntax")),
            self.assertRaises(vendor_deps.VendorPluginError),
        ):
            active_vendor()

    def test_explicit_vendor_that_cannot_load_is_fatal(self):
        with (
            patch.dict("os.environ", {"MONAI_VENDOR": "rocm"}),
            patch.object(vendor_deps, "_import_plugin", side_effect=SyntaxError("invalid syntax")),
            self.assertRaises(vendor_deps.VendorPluginError),
        ):
            active_vendor()


class TestPluginValidation(unittest.TestCase):
    """A plugin that imports cleanly but is malformed must not reach the rewrites.

    ``key in substitutions`` is a substring test on a string, so a non-mapping table would match
    nothing and silently emit the unmodified lists into this vendor's wheel.
    """

    def _load_with(self, **attributes):
        plugin = types.SimpleNamespace(SUBSTITUTIONS={}, extras_for=lambda name: [], extra_requirements=list)
        for key, value in attributes.items():
            setattr(plugin, key, value)
        with (
            patch.dict("os.environ", {"MONAI_VENDOR": "rocm"}),
            patch.object(vendor_deps, "_import_plugin", return_value=plugin),
        ):
            return active_vendor()

    def test_well_formed_plugin_loads(self):
        self.assertEqual("rocm", self._load_with().name)

    def test_substitutions_must_be_a_mapping(self):
        for table in ("not a dict", ["cucim-cu12"], None, 7):
            with self.subTest(table=table), self.assertRaises(vendor_deps.VendorPluginError):
                self._load_with(SUBSTITUTIONS=table)

    def test_required_callables_are_checked(self):
        for attribute in ("extras_for", "extra_requirements"):
            with self.subTest(attribute=attribute), self.assertRaises(vendor_deps.VendorPluginError):
                self._load_with(**{attribute: "not callable"})

    def test_substitution_keys_must_be_canonical(self):
        # Lookups canonicalise the requirement name, so "cucim_cu12" could never match "cucim-cu12".
        for key in ("cucim_cu12", "cucim.cu12", "CuCIM-cu12", 7):
            with self.subTest(key=key), self.assertRaises(vendor_deps.VendorPluginError):
                self._load_with(SUBSTITUTIONS={key: "amd-hipcim"})

    def test_canonical_substitution_keys_are_accepted(self):
        self.assertEqual("rocm", self._load_with(SUBSTITUTIONS={"cucim-cu12": "amd-hipcim"}).name)


class TestHardwareWithoutVendor(unittest.TestCase):
    """``pip`` build isolation can hand the build a generic torch while sitting on vendor hardware."""

    def test_warns_when_hardware_present_but_torch_is_not(self):
        probe = vendor_deps._Registration("vendor_rocm", lambda: False, lambda: True)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True),
            self.assertWarnsRegex(UserWarning, "rocm hardware is present"),
        ):
            self.assertIsNone(active_vendor())

    def test_silent_when_hardware_absent(self):
        probe = vendor_deps._Registration("vendor_rocm", lambda: False, lambda: False)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            self.assertIsNone(active_vendor())
        self.assertEqual([], [str(w.message) for w in caught])

    def test_silent_when_the_vendor_is_active(self):
        probe = vendor_deps._Registration("vendor_rocm", lambda: True, lambda: True)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(vendor_deps.REGISTRY, {"rocm": probe}, clear=True),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            self.assertEqual("rocm", active_vendor().name)
        self.assertEqual([], [str(w.message) for w in caught])

    def test_gpu_probe_ignores_the_cpu_node(self):
        with tempfile.TemporaryDirectory() as root:
            nodes = Path(root)
            (nodes / "0").mkdir()
            (nodes / "0" / "properties").write_text("cpu_cores_count 128\ngfx_target_version 0\n")
            with patch.object(vendor_deps, "KFD_TOPOLOGY_NODES", str(nodes)):
                self.assertFalse(vendor_deps._amd_gpu_present())
            (nodes / "1").mkdir()
            (nodes / "1" / "properties").write_text("cpu_cores_count 0\ngfx_target_version 90500\n")
            with patch.object(vendor_deps, "KFD_TOPOLOGY_NODES", str(nodes)):
                self.assertTrue(vendor_deps._amd_gpu_present())

    def test_gpu_probe_without_the_amdgpu_driver(self):
        with patch.object(vendor_deps, "KFD_TOPOLOGY_NODES", "/nonexistent-kfd"):
            self.assertFalse(vendor_deps._amd_gpu_present())


if __name__ == "__main__":
    unittest.main()
