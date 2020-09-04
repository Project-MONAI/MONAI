# Copyright 2020 MONAI Consortium
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
import warnings

from setuptools import find_packages, setup

import versioneer

# TODO: debug mode -g -O0, compile test cases

FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
SKIP_BUILD = os.getenv("SKIP_MONAI_BUILD", "0") == "1"

BUILD_CPP = BUILD_CUDA = False
try:
    import torch
    print(f"setup.py with torch {torch.__version__}")
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    BUILD_CPP = True
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    BUILD_CUDA = (torch.cuda.is_available() and (CUDA_HOME is not None)) or FORCE_CUDA
except ImportError:
    torch = CppExtension = BuildExtension = CUDA_HOME = CUDAExtension = None
    warnings.warn("extension build skipped.")
finally:
    if SKIP_BUILD:
        BUILD_CPP = BUILD_CUDA = False
    print(f"BUILD_MONAI_CPP={BUILD_CPP}, BUILD_MONAI_CUDA={BUILD_CUDA}")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(this_dir, "monai", "csrc")
    include_dirs = [ext_dir]

    main_src = glob.glob(os.path.join(ext_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(ext_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(ext_dir, "**", "*.cu"))

    extension = None
    define_macros = []
    extra_compile_args = {}
    sources = main_src + source_cpu
    if BUILD_CPP:
        extension = CppExtension
    if BUILD_CUDA:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args = {"cxx": [], "nvcc": []}
    if extension is None:
        return []  # compile nothing

    ext_modules = [
        extension(
            name="monai._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


def get_cmds():
    cmds = versioneer.get_cmdclass()

    if not (BUILD_CPP or BUILD_CUDA):
        return cmds

    cmds.update({"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)})
    return cmds


setup(
    version=versioneer.get_version(),
    cmdclass=get_cmds(),
    packages=find_packages(exclude=("docs", "examples", "tests", "research")),
    zip_safe=False,
    package_data={"monai": ["py.typed"]},
    ext_modules=get_extensions(),
)
