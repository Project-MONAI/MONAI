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
import unittest
from pathlib import Path
from unittest.mock import patch

from monai.config import vendor_rocm
from monai.config.vendor_deps import apply_to_dependencies, apply_to_optional_dependencies
from monai.config.vendor_rocm import (
    DEFAULT_GPU_ARCHS,
    DEFAULT_ROCM_SERIES,
    detect_gpu_archs,
    detect_rocm_series,
    extra_requirements,
    extras_for,
)


class TestReleaseFromDisk(unittest.TestCase):
    """Exercises the real filesystem parsing rather than mocking it out."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_pip_sdk_layout(self):
        # ROCM_PATH points into site-packages, where the `rocm` dist-info sits beside it.
        site = self.root / "site-packages"
        (site / "rocm-10.1.0.dist-info").mkdir(parents=True)
        (site / "_rocm_sdk_core").mkdir()
        with patch.dict("os.environ", {"ROCM_PATH": str(site / "_rocm_sdk_core")}, clear=True):
            self.assertEqual("10.1", vendor_rocm._rocm_release_from_disk())

    def test_pip_sdk_layout_ignores_sibling_rocm_packages(self):
        site = self.root / "site-packages"
        (site / "rocm_sdk_core-10.1.0.dist-info").mkdir(parents=True)
        (site / "rocm_bootstrap-0.3.0.dist-info").mkdir()
        (site / "rocm-10.0.0.dist-info").mkdir()
        (site / "_rocm_sdk_core").mkdir()
        with patch.dict("os.environ", {"ROCM_PATH": str(site / "_rocm_sdk_core")}, clear=True):
            self.assertEqual("10.0", vendor_rocm._rocm_release_from_disk())

    def test_classic_layout(self):
        rocm = self.root / "opt" / "rocm"
        (rocm / ".info").mkdir(parents=True)
        (rocm / ".info" / "version").write_text("10.0.0-18\n")
        with patch.dict("os.environ", {"ROCM_PATH": str(rocm)}, clear=True):
            self.assertEqual("10.0", vendor_rocm._rocm_release_from_disk())

    def test_rocm_home_is_consulted(self):
        rocm = self.root / "rocm"
        (rocm / ".info").mkdir(parents=True)
        (rocm / ".info" / "version").write_text("7.14.1\n")
        with patch.dict("os.environ", {"ROCM_HOME": str(rocm)}, clear=True):
            self.assertEqual("7.14", vendor_rocm._rocm_release_from_disk())

    def test_nothing_found(self):
        with patch.dict("os.environ", {"ROCM_PATH": str(self.root / "absent")}, clear=True):
            self.assertEqual("", vendor_rocm._rocm_release_from_disk())

    def test_feeds_detect_rocm_series(self):
        site = self.root / "site-packages"
        (site / "rocm-10.1.0.dist-info").mkdir(parents=True)
        (site / "_rocm_sdk_core").mkdir()
        with patch.dict("os.environ", {"ROCM_PATH": str(site / "_rocm_sdk_core")}, clear=True):
            self.assertEqual("10.1", detect_rocm_series())

    def test_system_install_does_not_shadow_this_environment(self):
        # A host /opt/rocm must not win over the SDK the build is actually using.
        system = self.root / "opt" / "rocm"
        (system / ".info").mkdir(parents=True)
        (system / ".info" / "version").write_text("7.2.3\n")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vendor_rocm, "SYSTEM_ROCM_ROOT", str(system)),
            patch.object(vendor_rocm, "_installed_rocm_version", return_value="10.1.0"),
        ):
            self.assertEqual("10.1", detect_rocm_series())

    def test_system_install_is_used_when_nothing_else_knows(self):
        system = self.root / "opt" / "rocm"
        (system / ".info").mkdir(parents=True)
        (system / ".info" / "version").write_text("7.2.3\n")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vendor_rocm, "SYSTEM_ROCM_ROOT", str(system)),
            patch.object(vendor_rocm, "_installed_rocm_version", return_value=""),
            patch.object(vendor_rocm, "_run", return_value=""),
        ):
            self.assertEqual("7.2", detect_rocm_series())


class TestDetection(unittest.TestCase):
    def test_series_from_env(self):
        with patch.dict("os.environ", {"MONAI_ROCM_SERIES": "7.14.1"}):
            self.assertEqual("7.14", detect_rocm_series())

    def test_series_fallback(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vendor_rocm, "_rocm_release_from_disk", return_value=""),
            patch.object(vendor_rocm, "_installed_rocm_version", return_value=""),
            patch.object(vendor_rocm, "SYSTEM_ROCM_ROOT", "/nonexistent-rocm"),
            patch.object(vendor_rocm, "_run", return_value=""),
        ):
            self.assertEqual(DEFAULT_ROCM_SERIES, detect_rocm_series())

    def test_series_from_disk(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vendor_rocm, "_rocm_release_from_disk", return_value="10.0"),
        ):
            self.assertEqual("10.0", detect_rocm_series())

    def test_series_from_rocm_sdk(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vendor_rocm, "_rocm_release_from_disk", return_value=""),
            patch.object(vendor_rocm, "_installed_rocm_version", return_value=""),
            patch.object(vendor_rocm, "SYSTEM_ROCM_ROOT", "/nonexistent-rocm"),
            patch.object(vendor_rocm, "_run", return_value="10.1.0\n"),
        ):
            self.assertEqual("10.1", detect_rocm_series())

    def test_series_never_uses_hip_version(self):
        # ROCm releases and HIP carry different numbers; pinning `rocm` to HIP matches no index.
        commands: list[list[str]] = []

        def record(cmd):
            commands.append(cmd)
            return ""

        with (
            patch.dict("os.environ", {"HIP_VERSION": "7.16.26385"}, clear=True),
            patch.object(vendor_rocm, "_rocm_release_from_disk", return_value=""),
            patch.object(vendor_rocm, "_installed_rocm_version", return_value=""),
            patch.object(vendor_rocm, "SYSTEM_ROCM_ROOT", "/nonexistent-rocm"),
            patch.object(vendor_rocm, "_run", side_effect=record),
        ):
            self.assertEqual(DEFAULT_ROCM_SERIES, detect_rocm_series())
        # rocm-sdk reports the release; hipcc reports HIP and is never asked.
        self.assertEqual([["rocm-sdk", "version"]], commands)

    def test_gpu_archs_from_env(self):
        with patch.dict("os.environ", {"GPU_TARGETS": "gfx90a;GFX942;gfx942"}):
            self.assertEqual(["gfx90a", "gfx942"], detect_gpu_archs())

    def test_gpu_archs_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(list(DEFAULT_GPU_ARCHS), detect_gpu_archs())


class TestPluginContract(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"MONAI_ROCM_SERIES": "10.0", "GPU_TARGETS": "gfx942"})
        env.start()
        self.addCleanup(env.stop)

    def test_extras_only_for_torch(self):
        self.assertEqual(["device-gfx942"], extras_for("torch"))
        self.assertEqual([], extras_for("numpy"))

    def test_rocm_requirement(self):
        self.assertEqual(["rocm[libraries,devel,device-gfx942]>=10.0.0a0,<10.2"], extra_requirements())

    def test_dependencies(self):
        self.assertEqual(
            ["torch[device-gfx942]>=2.8.0", "numpy>=1.24,<3.0", "rocm[libraries,devel,device-gfx942]>=10.0.0a0,<10.2"],
            apply_to_dependencies(["torch>=2.8.0", "numpy>=1.24,<3.0"], vendor_rocm),
        )

    def test_cucim_variants_collapse(self):
        extras = {
            "cucim": [
                "cucim-cu12; platform_system == 'Linux' and python_version <= '3.10'",
                "cucim-cu13; platform_system == 'Linux' and python_version >= '3.11'",
            ]
        }
        self.assertEqual(
            {"cucim": ["amd-hipcim>=26.6.0; platform_system == 'Linux'"]},
            apply_to_optional_dependencies(extras, vendor_rocm),
        )

    def test_cupy_and_dropped_packages(self):
        extras = {
            "cupy": ["cupy-cuda13x!=14.1.0"],
            "nvimgcodec": ["pydicom", "cupy-cuda13x[ctk]!=14.1.0", "nvidia-nvimgcodec-cu13[all]>=0.8.0"],
            "pynvml": ["nvidia-ml-py"],
            "nni": ["nni; platform_system == 'Linux'", "filelock<3.12.0"],
        }
        self.assertEqual(
            {
                "cupy": ["amd-cupy; platform_system == 'Linux'"],
                "nvimgcodec": ["pydicom", "amd-cupy; platform_system == 'Linux'"],
                "pynvml": [],
                "nni": ["filelock<3.12.0"],
            },
            apply_to_optional_dependencies(extras, vendor_rocm),
        )


if __name__ == "__main__":
    unittest.main()
