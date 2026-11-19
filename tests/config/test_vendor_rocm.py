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

import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from monai.config import vendor_rocm
from monai.config.vendor_deps import REGISTRY
from monai.config.vendor_rocm import DEFAULT_GPU_ARCHS, DEFAULT_ROCM_SERIES, RocmVendor, _release_under

ROCM = RocmVendor()


@contextmanager
def rocm_sources(env=None, installed=None, sdk_output=None, system_root="/nonexistent-rocm"):
    """Control every place ``rocm_series`` looks, yielding the list of commands it ran."""
    commands: list[list[str]] = []

    def version(name):
        if installed is None:
            raise metadata.PackageNotFoundError(name)
        return installed

    def run(cmd, **kwargs):
        commands.append(cmd)
        if sdk_output is None:
            raise FileNotFoundError(cmd[0])
        return types.SimpleNamespace(stdout=sdk_output)

    with (
        patch.dict("os.environ", env or {}, clear=True),
        patch.object(vendor_rocm.metadata, "version", side_effect=version),
        patch.object(vendor_rocm.subprocess, "run", side_effect=run),
        patch.object(vendor_rocm, "SYSTEM_ROCM_ROOT", system_root),
    ):
        yield commands


class TestReleaseUnder(unittest.TestCase):
    """Exercises the real filesystem parsing rather than mocking it out."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def pip_sdk(self, *dist_infos):
        site = self.root / "site-packages"
        for name in (*dist_infos, "_rocm_sdk_core"):
            (site / name).mkdir(parents=True)
        return str(site / "_rocm_sdk_core")

    def classic(self, version):
        rocm = self.root / "opt" / "rocm"
        (rocm / ".info").mkdir(parents=True)
        (rocm / ".info" / "version").write_text(version)
        return str(rocm)

    def test_pip_sdk_layout(self):
        # ROCM_PATH points into site-packages, where the `rocm` dist-info sits beside it.
        self.assertEqual("10.1", _release_under(self.pip_sdk("rocm-10.1.0.dist-info")))

    def test_pip_sdk_layout_ignores_sibling_rocm_packages(self):
        root = self.pip_sdk("rocm_sdk_core-10.1.0.dist-info", "rocm_bootstrap-0.3.0.dist-info", "rocm-10.0.0.dist-info")
        self.assertEqual("10.0", _release_under(root))

    def test_classic_layout(self):
        self.assertEqual("10.0", _release_under(self.classic("10.0.0-18\n")))

    def test_nothing_found(self):
        self.assertEqual("", _release_under(str(self.root / "absent")))
        self.assertEqual("", _release_under(None))

    def test_rocm_path_feeds_series(self):
        with rocm_sources(env={"ROCM_PATH": self.pip_sdk("rocm-10.1.0.dist-info")}):
            self.assertEqual("10.1", ROCM.rocm_series())

    def test_rocm_home_feeds_series(self):
        with rocm_sources(env={"ROCM_HOME": self.classic("7.14.1\n")}):
            self.assertEqual("7.14", ROCM.rocm_series())

    def test_system_install_does_not_shadow_this_environment(self):
        # A host /opt/rocm must not win over the SDK the build is actually using.
        with rocm_sources(installed="10.1.0", system_root=self.classic("7.2.3\n")):
            self.assertEqual("10.1", ROCM.rocm_series())

    def test_system_install_is_used_when_nothing_else_knows(self):
        with rocm_sources(system_root=self.classic("7.2.3\n")):
            self.assertEqual("7.2", ROCM.rocm_series())


class TestDetection(unittest.TestCase):
    def test_series_from_env_wins(self):
        with rocm_sources(env={"MONAI_ROCM_SERIES": "7.14.1"}, installed="10.1.0", sdk_output="10.1.0\n"):
            self.assertEqual("7.14", ROCM.rocm_series())

    def test_series_from_installed_package(self):
        with rocm_sources(installed="10.1.0"):
            self.assertEqual("10.1", ROCM.rocm_series())

    def test_series_from_rocm_sdk(self):
        with rocm_sources(sdk_output="10.1.0\n"):
            self.assertEqual("10.1", ROCM.rocm_series())

    def test_series_fallback(self):
        with rocm_sources():
            self.assertEqual(DEFAULT_ROCM_SERIES, ROCM.rocm_series())

    def test_series_never_uses_hip_version(self):
        # ROCm releases and HIP carry different numbers; pinning `rocm` to HIP matches no index.
        with rocm_sources(env={"HIP_VERSION": "7.16.26385"}, sdk_output="") as commands:
            self.assertEqual(DEFAULT_ROCM_SERIES, ROCM.rocm_series())
        # rocm-sdk reports the release; hipcc reports HIP and is never asked.
        self.assertEqual([["rocm-sdk", "version"]], commands)

    def test_gpu_archs_from_env(self):
        with patch.dict("os.environ", {"GPU_TARGETS": "gfx90a;GFX942;gfx942"}):
            self.assertEqual(["gfx90a", "gfx942"], ROCM.gpu_archs())

    def test_gpu_archs_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(list(DEFAULT_GPU_ARCHS), ROCM.gpu_archs())

    def test_detect_follows_torch_version_hip(self):
        for hip, expected in (("7.16.26385", True), (None, False)):
            torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=hip, cuda=None))
            with self.subTest(hip=hip), patch.dict(sys.modules, {"torch": torch}):
                self.assertEqual(expected, ROCM.detect())

    def test_detect_without_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            self.assertFalse(ROCM.detect())

    def test_hardware_probe_ignores_the_cpu_node(self):
        with tempfile.TemporaryDirectory() as root:
            nodes = Path(root)
            (nodes / "0").mkdir()
            (nodes / "0" / "properties").write_text("cpu_cores_count 128\ngfx_target_version 0\n")
            with patch.object(vendor_rocm, "KFD_TOPOLOGY_NODES", str(nodes)):
                self.assertFalse(ROCM.hardware_present())
            (nodes / "1").mkdir()
            (nodes / "1" / "properties").write_text("cpu_cores_count 0\ngfx_target_version 90500\n")
            with patch.object(vendor_rocm, "KFD_TOPOLOGY_NODES", str(nodes)):
                self.assertTrue(ROCM.hardware_present())

    def test_hardware_probe_without_the_amdgpu_driver(self):
        with patch.object(vendor_rocm, "KFD_TOPOLOGY_NODES", "/nonexistent-kfd"):
            self.assertFalse(ROCM.hardware_present())


class TestRewrites(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"MONAI_ROCM_SERIES": "10.0", "GPU_TARGETS": "gfx942"})
        env.start()
        self.addCleanup(env.stop)

    def test_registered_on_import(self):
        self.assertIsInstance(REGISTRY["rocm"], RocmVendor)

    def test_extras_only_for_torch(self):
        self.assertEqual(["device-gfx942"], ROCM.extras_for("torch"))
        self.assertEqual([], ROCM.extras_for("numpy"))

    def test_rocm_requirement(self):
        self.assertEqual(["rocm[libraries,devel,device-gfx942]>=10.0.0a0,<10.2"], ROCM.extra_requirements())

    def test_dependencies(self):
        self.assertEqual(
            ["torch[device-gfx942]>=2.8.0", "numpy>=1.24,<3.0", "rocm[libraries,devel,device-gfx942]>=10.0.0a0,<10.2"],
            ROCM.apply_to_dependencies(["torch>=2.8.0", "numpy>=1.24,<3.0"]),
        )

    def test_cucim_variants_collapse(self):
        extras = {
            "cucim": [
                "cucim-cu12; platform_system == 'Linux' and python_version <= '3.10'",
                "cucim-cu13; platform_system == 'Linux' and python_version >= '3.11'",
            ]
        }
        self.assertEqual(
            {"cucim": ["amd-hipcim>=26.6.0; platform_system == 'Linux'"]}, ROCM.apply_to_optional_dependencies(extras)
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
            ROCM.apply_to_optional_dependencies(extras),
        )


if __name__ == "__main__":
    unittest.main()
