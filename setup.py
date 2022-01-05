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

import glob
import os
import re
import sys
import warnings

import pkg_resources
from setuptools import find_packages, setup

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

    BUILD_CUDA = (CUDA_HOME is not None) if torch.cuda.is_available() else FORCE_CUDA

    _pt_version = pkg_resources.parse_version(torch.__version__).release  # type: ignore[attr-defined]
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
        extra_compile_args = {"cxx": [], "nvcc": []}
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
    if extension is None or not sources:
        return []  # compile nothing

    ext_modules = [
        extension(
            name="monai._C",
            sources=sources,
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


# Gathering source used for JIT extensions to include in package_data.
jit_extension_source = []

for ext in ["cpp", "cu", "h", "cuh"]:
    glob_path = os.path.join("monai", "_extensions", "**", f"*.{ext}")
    jit_extension_source += glob.glob(glob_path, recursive=True)

jit_extension_source = [os.path.join("..", path) for path in jit_extension_source]

setup(
    version=versioneer.get_version(),
    cmdclass=get_cmds(),
    packages=find_packages(exclude=("docs", "examples", "tests")),
    zip_safe=False,
    package_data={"monai": ["py.typed", *jit_extension_source]},
    ext_modules=get_extensions(),
)
