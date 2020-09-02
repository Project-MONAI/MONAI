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
    ext_modules = []
    if BUILD_CPP:
        ext_modules.append(CppExtension("monai._C", ["monai/extensions/lltm/lltm.cpp"]))
    if BUILD_CUDA:
        ext_modules.append(
            CUDAExtension(
                "monai._C_CUDA", ["monai/extensions/lltm/lltm_cuda.cpp", "monai/extensions/lltm/lltm_cuda_kernel.cu"],
            )
        )
    return ext_modules


def get_cmds():
    cmds = versioneer.get_cmdclass()
    if BUILD_CPP or BUILD_CUDA:
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
