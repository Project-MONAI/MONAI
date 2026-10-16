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
AMD ROCm plugin for :mod:`monai.config.vendor_deps`.

AMD publishes drop-in replacements for the NVIDIA imaging stack under different distribution names:
``amd-hipcim`` ships the ``cucim`` Python namespace and ``amd-cupy`` ships ``cupy``, so MONAI's
``optional_import("cucim"...)`` / ``optional_import("cupy")`` call sites work unchanged once they
are installed.

Required dependencies additionally get the AMD GPU device extras (``torch[device-gfx942]``) and a
``rocm`` SDK requirement, which is what steers pip towards the ROCm build of PyTorch.

Environment variables:

* ``MONAI_ROCM_SERIES``: ROCm release ``MAJOR.MINOR`` to pin, bypassing autodetection.
* ``GPU_TARGETS`` / ``AMDGPU_TARGETS``: semicolon- or comma-separated AMD GPU architectures.

This module is imported only when :data:`monai.config.vendor_deps.REGISTRY` detects ROCm, so it
never runs during a build for another vendor.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

NAME = "rocm"

# ROCm release series (MAJOR.MINOR) to pin against when it cannot be detected. This tracks the ROCm
# *release* -- the version of the `rocm` pip package, e.g. 10.0.x -- NOT the HIP version, which is a
# separate internal number (ROCm 10.0 ships HIP 7.15, ROCm 10.1 ships HIP 7.16).
DEFAULT_ROCM_SERIES = "10.0"
DEFAULT_GPU_ARCHS = ("gfx942", "gfx950")

# Classic non-pip ROCm install, consulted only after the SDK in the current environment.
SYSTEM_ROCM_ROOT = "/opt/rocm"

# How many minor series beyond the detected one the pin accepts. Artifacts built against one ROCm
# minor have been verified to run on the next: amd-cupy/amd-hipcim/amd-monai built for 10.0 pass on
# a 10.1 runtime, including hiprtc JIT, hipFFT/hipBLAS/hipRAND and the cucim.skimage kernels.
ROCM_SERIES_SPAN = 2

# A value of None drops the requirement entirely: nvidia-ml-py raises at import time without an
# NVIDIA driver and nni depends on it; the nvidia-nvimgcodec wheels have no ROCm equivalent.
_HIPCIM = "amd-hipcim>=26.6.0; platform_system == 'Linux'"
_CUPY = "amd-cupy; platform_system == 'Linux'"

SUBSTITUTIONS: dict[str, str | None] = {
    "cucim-cu12": _HIPCIM,
    "cucim-cu13": _HIPCIM,
    "cupy-cuda12x": _CUPY,
    "cupy-cuda13x": _CUPY,
    "nvidia-nvimgcodec-cu12": None,
    "nvidia-nvimgcodec-cu13": None,
    "nvidia-ml-py": None,
    "nni": None,
}


def _run(cmd: list[str]) -> str:
    """stdout of ``cmd``, or "" if it is missing or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def _installed_rocm_version() -> str:
    """Version of the ``rocm`` package installed in this environment, or "" if absent."""
    try:
        from importlib.metadata import version

        return version("rocm")
    except Exception:
        return ""


def _release_under(root: Path) -> str:
    """ROCm release ``MAJOR.MINOR`` recorded under ``root``, or "" if not found there.

    Handles the pip ROCm SDK, whose release is the ``rocm`` package version in the ``*.dist-info``
    beside ``root``'s ``_rocm_sdk_*`` tree, and a classic install, whose release is in
    ``.info/version``.
    """
    for dist in sorted(root.parent.glob("rocm-*.dist-info")):
        match = re.search(r"rocm-(\d+\.\d+)", dist.name)
        if match:
            return match.group(1)
    try:
        text = (root / ".info" / "version").read_text()
    except OSError:
        return ""
    match = re.search(r"(\d+\.\d+)", text)
    return match.group(1) if match else ""


def _rocm_release_from_disk() -> str:
    """ROCm release read from the tree ``ROCM_PATH`` / ``ROCM_HOME`` point at, or "" if unset.

    Reads files only, so it still works under ``pip`` build isolation where the PATH-based
    ``rocm-sdk`` console script may not be reachable. Mirrors hipCIM's detection.
    """
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME")):
        if root:
            release = _release_under(Path(root))
            if release:
                return release
    return ""


def detect_rocm_series() -> str:
    """
    ROCm release ``MAJOR.MINOR`` to pin against. First match wins: ``MONAI_ROCM_SERIES``, the tree
    ``ROCM_PATH``/``ROCM_HOME`` point at, the ``rocm`` package installed here, ``rocm-sdk version``,
    a system install under :data:`SYSTEM_ROCM_ROOT`, else :data:`DEFAULT_ROCM_SERIES`.

    Two orderings matter. The HIP version is never consulted: ``torch.version.hip`` and
    ``hipcc --version`` report HIP, a different number from the ROCm release the ``rocm`` package is
    versioned by, so pinning to it yields a requirement no index can satisfy. And a system install
    is consulted last, so an unrelated ``/opt/rocm`` cannot shadow the SDK being built against.
    """
    for text in (
        os.environ.get("MONAI_ROCM_SERIES", ""),
        _rocm_release_from_disk(),
        _installed_rocm_version(),
        _run(["rocm-sdk", "version"]),
        _release_under(Path(SYSTEM_ROCM_ROOT)),
    ):
        match = re.search(r"(\d+\.\d+)", text)
        if match:
            return match.group(1)
    return DEFAULT_ROCM_SERIES


def detect_gpu_archs() -> list[str]:
    """
    AMD GPU architectures this build targets, used both for the ``torch[device-gfx...]`` extras and
    for the ``rocm[device-gfx...]`` features. Read from ``GPU_TARGETS`` or ``AMDGPU_TARGETS``,
    falling back to :data:`DEFAULT_GPU_ARCHS`.
    """
    raw = os.environ.get("GPU_TARGETS") or os.environ.get("AMDGPU_TARGETS", "")
    archs: list[str] = []
    for match in re.findall(r"gfx[0-9a-f]+", raw, re.IGNORECASE):
        arch = match.lower()
        if arch not in archs:
            archs.append(arch)
    return archs or list(DEFAULT_GPU_ARCHS)


def extras_for(name: str) -> list[str]:
    """Device selector extras for ``name``; only ``torch`` carries them."""
    if name != "torch":
        return []
    return [f"device-{arch}" for arch in detect_gpu_archs()]


def extra_requirements() -> list[str]:
    """The ``rocm`` SDK requirement: runtime libraries, the development tree and a device feature per
    targeted GPU arch, spanning :data:`ROCM_SERIES_SPAN` minor series from the detected one.

    ``devel`` is required after install, not just to build: ``monai/_extensions`` JIT-compiles its
    HIP sources on first use through ``torch.utils.cpp_extension.load()``, which needs ``hipcc`` and
    the ROCm headers. amd-cupy likewise JIT-compiles kernels through ``hipcc``. Whether the compiler
    resolves without ``devel`` depends on where ``ROCM_PATH`` points -- it is in ``_rocm_sdk_core``
    on ROCm 10.x -- so do not drop it on the strength of a core-only layout happening to work.
    """
    major, minor = (int(part) for part in detect_rocm_series().split(".")[:2])
    features = ["libraries", "devel"] + [f"device-{arch}" for arch in detect_gpu_archs()]
    return [f"rocm[{','.join(features)}]>={major}.{minor}.0a0,<{major}.{minor + ROCM_SERIES_SPAN}"]
