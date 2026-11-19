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

"""
AMD ROCm vendor for :mod:`monai.config.vendor_deps`.

AMD publishes drop-in replacements for the NVIDIA imaging stack under different distribution names:
``amd-hipcim`` ships the ``cucim`` Python namespace and ``amd-cupy`` ships ``cupy``, so MONAI's
``optional_import("cucim"...)`` / ``optional_import("cupy")`` call sites work unchanged once they
are installed.

Required dependencies additionally get the AMD GPU device extras (``torch[device-gfx942]``) and a
``rocm`` SDK requirement, which is what steers pip towards the ROCm build of PyTorch.

Environment variables:

* ``MONAI_ROCM_SERIES``: ROCm release ``MAJOR.MINOR`` to pin, bypassing autodetection.
* ``GPU_TARGETS`` / ``AMDGPU_TARGETS``: semicolon- or comma-separated AMD GPU architectures.
"""

from __future__ import annotations

import os
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from .vendor_deps import Vendor, register
else:  # imported directly by setup.py or print_dependencies.py, outside the package
    from vendor_deps import Vendor, register

# ROCm release series (MAJOR.MINOR) to pin against when it cannot be detected. This tracks the ROCm
# *release* -- the version of the `rocm` pip package, e.g. 10.0.x -- NOT the HIP version, which is a
# separate internal number (ROCm 10.0 ships HIP 7.15, ROCm 10.1 ships HIP 7.16).
DEFAULT_ROCM_SERIES = "10.0"
DEFAULT_GPU_ARCHS = ("gfx942", "gfx950")

# Classic non-pip ROCm install, consulted only after the SDK in the current environment.
SYSTEM_ROCM_ROOT = "/opt/rocm"

# Compute nodes exposed by the amdgpu driver; node 0 is the host CPU.
KFD_TOPOLOGY_NODES = "/sys/class/kfd/kfd/topology/nodes"

# How many minor series beyond the detected one the pin accepts. Artifacts built against one ROCm
# minor have been verified to run on the next: amd-cupy/amd-hipcim/amd-monai built for 10.0 pass on
# a 10.1 runtime, including hiprtc JIT, hipFFT/hipBLAS/hipRAND and the cucim.skimage kernels.
ROCM_SERIES_SPAN = 2

_HIPCIM = "amd-hipcim>=26.6.0; platform_system == 'Linux'"
_CUPY = "amd-cupy; platform_system == 'Linux'"


def _release_under(root: str | None) -> str:
    """ROCm release ``MAJOR.MINOR`` recorded under the install tree ``root``, or "" if not found there.

    Handles the pip ROCm SDK, whose release is the ``rocm`` package version in the ``*.dist-info``
    beside ``root``'s ``_rocm_sdk_*`` tree, and a classic install, whose release is in
    ``.info/version``. Only files are read, so this works under pip build isolation.
    """
    if not root:
        return ""
    for dist in sorted(Path(root).parent.glob("rocm-*.dist-info")):
        match = re.search(r"rocm-(\d+\.\d+)", dist.name)
        if match:
            return match.group(1)
    try:
        match = re.search(r"(\d+\.\d+)", Path(root, ".info", "version").read_text())
    except OSError:
        return ""
    return match.group(1) if match else ""


class RocmVendor(Vendor):
    """Swaps the NVIDIA imaging stack for AMD's and pins the ROCm SDK the build was made against."""

    name = "rocm"

    # None drops the requirement entirely: nvidia-ml-py raises at import time without an NVIDIA
    # driver and nni depends on it; the nvidia-nvimgcodec wheels have no ROCm equivalent.
    substitutions = {
        "cucim-cu12": _HIPCIM,
        "cucim-cu13": _HIPCIM,
        "cupy-cuda12x": _CUPY,
        "cupy-cuda13x": _CUPY,
        "nvidia-nvimgcodec-cu12": None,
        "nvidia-nvimgcodec-cu13": None,
        "nvidia-ml-py": None,
        "nni": None,
    }

    def detect(self) -> bool:
        """True when the PyTorch this build sees is a ROCm build."""
        try:
            import torch

            return bool(getattr(torch.version, "hip", None))
        except Exception:  # torch absent, or present but not importable in this environment
            return False

    def hardware_present(self) -> bool:
        """True when the amdgpu driver exposes a GPU compute node."""
        try:
            nodes = list(os.scandir(KFD_TOPOLOGY_NODES))
        except OSError:
            return False
        for node in nodes:
            try:
                text = Path(node.path, "properties").read_text()
            except OSError:
                continue
            match = re.search(r"^gfx_target_version (\d+)", text, re.M)
            if match and match.group(1) != "0":
                return True
        return False

    def gpu_archs(self) -> list[str]:
        """AMD GPU architectures this build targets, from ``GPU_TARGETS`` or ``AMDGPU_TARGETS``, falling back to
        :data:`DEFAULT_GPU_ARCHS`. Used for both the ``torch[device-gfx...]`` and ``rocm[device-gfx...]`` extras.
        """
        raw = os.environ.get("GPU_TARGETS") or os.environ.get("AMDGPU_TARGETS", "")
        archs: list[str] = []
        for match in re.findall(r"gfx[0-9a-f]+", raw, re.IGNORECASE):
            if match.lower() not in archs:
                archs.append(match.lower())
        return archs or list(DEFAULT_GPU_ARCHS)

    def rocm_series(self) -> str:
        """
        ROCm release ``MAJOR.MINOR`` to pin against. First match wins: ``MONAI_ROCM_SERIES``, the tree
        ``ROCM_PATH``/``ROCM_HOME`` point at, the ``rocm`` package installed here, ``rocm-sdk version``,
        a system install under :data:`SYSTEM_ROCM_ROOT`, else :data:`DEFAULT_ROCM_SERIES`.

        Two orderings matter. The HIP version is never consulted: ``torch.version.hip`` and
        ``hipcc --version`` report HIP, a different number from the ROCm release the ``rocm`` package is
        versioned by, so pinning to it yields a requirement no index can satisfy. And a system install
        is consulted last, so an unrelated ``/opt/rocm`` cannot shadow the SDK being built against.
        """
        candidates = [os.environ.get("MONAI_ROCM_SERIES", "")]
        candidates += [_release_under(os.environ.get(variable)) for variable in ("ROCM_PATH", "ROCM_HOME")]
        try:
            candidates.append(metadata.version("rocm"))
        except metadata.PackageNotFoundError:
            pass
        try:
            sdk = subprocess.run(["rocm-sdk", "version"], capture_output=True, text=True, check=False)
            candidates.append(sdk.stdout)
        except OSError:  # rocm-sdk is not on PATH
            pass
        candidates.append(_release_under(SYSTEM_ROCM_ROOT))

        for text in candidates:
            match = re.search(r"(\d+\.\d+)", text)
            if match:
                return match.group(1)
        return DEFAULT_ROCM_SERIES

    def extras_for(self, name: str) -> list[str]:
        """Device selector extras, which only ``torch`` carries."""
        return [f"device-{arch}" for arch in self.gpu_archs()] if name == "torch" else []

    def extra_requirements(self) -> list[str]:
        """The ``rocm`` SDK requirement: runtime libraries, the development tree and a device feature per
        targeted GPU arch, spanning :data:`ROCM_SERIES_SPAN` minor series from the detected one.

        ``devel`` is required after install, not just to build: ``monai/_extensions`` JIT-compiles its
        HIP sources on first use through ``torch.utils.cpp_extension.load()``, which needs ``hipcc`` and
        the ROCm headers. amd-cupy likewise JIT-compiles kernels through ``hipcc``. Whether the compiler
        resolves without ``devel`` depends on where ``ROCM_PATH`` points -- it is in ``_rocm_sdk_core``
        on ROCm 10.x -- so do not drop it on the strength of a core-only layout happening to work.
        """
        major, minor = (int(part) for part in self.rocm_series().split("."))
        features = ["libraries", "devel"] + [f"device-{arch}" for arch in self.gpu_archs()]
        return [f"rocm[{','.join(features)}]>={major}.{minor}.0a0,<{major}.{minor + ROCM_SERIES_SPAN}"]


register(RocmVendor())
