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

import glob
import importlib.util
import os
import re
import sys
import warnings
from typing import Any, cast

from packaging import version
from setuptools import find_packages, setup
from setuptools.dist import Distribution

import versioneer

# TODO: debug mode -g -O0, compile test cases

RUN_BUILD = os.getenv("BUILD_MONAI", "0") == "1"
FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"  # flag ignored if BUILD_MONAI is False

BUILD_CPP = BUILD_CUDA = False
TORCH_VERSION = 0

try:
    import torch

    print(f"setup.py with torch {torch.__version__}")
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    BUILD_CPP = True
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    # On a ROCm build of torch, `CUDA_HOME` is None and the toolkit is located by `ROCM_HOME`
    # instead; `CUDAExtension` hipifies the .cu sources transparently in that case. Accept
    # either so the extensions are not silently skipped on ROCm.
    _toolkit_home = CUDA_HOME or getattr(torch.utils.cpp_extension, "ROCM_HOME", None)
    BUILD_CUDA = FORCE_CUDA or (torch.cuda.is_available() and (_toolkit_home is not None))

    _pt_version = version.parse(torch.__version__).release
    if _pt_version is None or len(_pt_version) < 3:
        raise AssertionError("unknown torch version")
    TORCH_VERSION = int(_pt_version[0]) * 10000 + int(_pt_version[1]) * 100 + int(_pt_version[2])
except (ImportError, TypeError, AssertionError, AttributeError) as e:
    warnings.warn(f"extension build skipped: {e}")
finally:
    if not RUN_BUILD:
        BUILD_CPP = BUILD_CUDA = False
        print("Please set environment variable `BUILD_MONAI=1` to enable Cpp/CUDA extension build.")
    print(f"BUILD_MONAI_CPP={BUILD_CPP}, BUILD_MONAI_CUDA={BUILD_CUDA}, TORCH_VERSION={TORCH_VERSION}.")


def torch_parallel_backend():
    try:
        match = re.search("^ATen parallel backend: (?P<backend>.*)$", torch._C._parallel_info(), re.MULTILINE)
        if match is None:
            return None
        backend = match.group("backend")
        if backend == "OpenMP":
            return "AT_PARALLEL_OPENMP"
        if backend == "native thread pool":
            return "AT_PARALLEL_NATIVE"
        if backend == "native thread pool and TBB":
            return "AT_PARALLEL_NATIVE_TBB"
    except (NameError, AttributeError):  # no torch or no binaries
        warnings.warn("Could not determine torch parallel_info.")
    return None


def omp_flags():
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin":
        # https://stackoverflow.com/questions/37362414/
        # return ["-fopenmp=libiomp5"]
        return []
    return ["-fopenmp"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(this_dir, "monai", "csrc")
    include_dirs = [ext_dir]

    source_cpu = glob.glob(os.path.join(ext_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(ext_dir, "**", "*.cu"), recursive=True)

    extension = None
    define_macros = [(f"{torch_parallel_backend()}", 1), ("MONAI_TORCH_VERSION", TORCH_VERSION)]
    extra_compile_args = {}
    extra_link_args = []
    sources = source_cpu
    if BUILD_CPP:
        extension = CppExtension
        extra_compile_args.setdefault("cxx", [])
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
        extra_link_args = omp_flags()
    if BUILD_CUDA:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        # Embed the maximum compute capability from TORCH_CUDA_ARCH_LIST
        _torch_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        _max_cc = 0
        if _torch_cuda_arch_list:
            for _maj, _min in re.findall(r"([0-9]+)\.([0-9]+)", _torch_cuda_arch_list):
                try:
                    _cc = int(_maj) * 100 + int(_min)
                    if _cc > _max_cc:
                        _max_cc = _cc
                except ValueError:
                    pass
        if _max_cc > 0:
            define_macros += [("MONAI_MAX_COMPUTE_CAPABILITY", _max_cc)]
        extra_compile_args = {"cxx": [], "nvcc": []}
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
    if extension is None or not sources:
        return []  # compile nothing

    ext_modules = [
        extension(
            name="monai._C",
            sources=list(map(os.path.relpath, sources)),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    return ext_modules


def get_cmds():
    cmds = versioneer.get_cmdclass()

    if not (BUILD_CPP or BUILD_CUDA):
        return cmds

    cmds.update({"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)})
    return cmds


def load_vendor_deps():
    """Import ``monai/config/vendor_deps.py`` by path, without importing the ``monai`` package."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monai", "config", "vendor_deps.py")
    spec = importlib.util.spec_from_file_location("monai_vendor_deps", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves a class's module through sys.modules
    spec.loader.exec_module(module)
    return module


def vendor_distclass():
    """``Distribution`` subclass rewriting the dependency metadata for the detected accelerator
    vendor, or None when there is none so the build keeps stock setuptools behaviour.
    """
    try:
        vendor_deps = load_vendor_deps()
        vendor = vendor_deps.active_vendor()
    except Exception as e:
        # A vendor that was detected but could not be loaded is fatal: emitting another vendor's
        # packages in this vendor's wheel is worse than failing the build.
        if os.environ.get("MONAI_VENDOR") or type(e).__name__ == "VendorPluginError":
            raise
        warnings.warn(f"accelerator vendor detection skipped: {e}")
        return None
    if vendor is None:
        return None

    class VendorDistribution(Distribution):
        # pyproject.toml wins over anything passed to setup(), so the lists can only be adjusted once
        # setuptools has applied them, and _finalize_requires() has to be re-run to propagate the
        # result into the wheel metadata.
        def parse_config_files(self, *args, **kwargs):
            super().parse_config_files(*args, **kwargs)
            self.install_requires = vendor.apply_to_dependencies(list(self.install_requires or []))
            self.extras_require = vendor.apply_to_optional_dependencies(dict(self.extras_require or {}))
            self._finalize_requires()

    return VendorDistribution


# Gathering source used for JIT extensions to include in package_data.
jit_extension_source = []

for ext in ["cpp", "cu", "h", "cuh"]:
    glob_path = os.path.join("monai", "_extensions", "**", f"*.{ext}")
    jit_extension_source += glob.glob(glob_path, recursive=True)

jit_extension_source = [os.path.join("..", path) for path in jit_extension_source]

_distclass = vendor_distclass()

setup(
    version=versioneer.get_version(),
    cmdclass=get_cmds(),
    **({"distclass": _distclass} if _distclass is not None else {}),
    packages=find_packages(exclude=("docs", "examples", "tests", "tests.*")),
    zip_safe=False,
    package_data=cast(Any, {"monai": ["py.typed", *jit_extension_source]}),
    ext_modules=get_extensions(),
)
