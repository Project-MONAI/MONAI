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

FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"

BUILD_CPP = BUILD_CUDA = False
try:
    import torch
    from torch.utils.cpp_extension import CppExtension

    BUILD_CPP = True
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    BUILD_CUDA = (torch.cuda.is_available() and (CUDA_HOME is not None)) or FORCE_CUDA
except ImportError:
    warnings.warn("torch extension build skipped.")
finally:
    print(f"BUILD_CPP={BUILD_CPP}, BUILD_CUDA={BUILD_CUDA}")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(this_dir, "monai", "extensions")
    ext_modules = []
    if BUILD_CPP:
        lltm_cpu = glob.glob(os.path.join(ext_dir, "lltm", "*_cpu.cpp"))
        ext_modules.append(CppExtension("monai._C", lltm_cpu))
    if BUILD_CUDA:
        lltm_gpu = glob.glob(os.path.join(ext_dir, "lltm", "*_cuda*"))
        ext_modules.append(CUDAExtension("monai._C_CUDA", lltm_gpu))
    return ext_modules


def get_cmds():
    cmds = versioneer.get_cmdclass()
    if not (BUILD_CPP or BUILD_CUDA):
        return cmds

    from torch.utils.cpp_extension import BuildExtension

    cmds.update({"build_ext": BuildExtension})
    return cmds


setup(
    version=versioneer.get_version(),
    cmdclass=get_cmds(),
    packages=find_packages(exclude=("docs", "examples", "tests", "research")),
    zip_safe=False,
    package_data={"monai": ["py.typed"]},
    ext_modules=get_extensions(),
)
