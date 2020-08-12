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

import warnings

from setuptools import find_packages, setup

import versioneer


def get_extensions():

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

        print(f"setup.py with torch {torch.__version__}")
    except ImportError:
        warnings.warn("torch cpp/cuda building skipped.")
        return []

    ext_modules = [CppExtension("monai._C", ["monai/networks/extensions/lltm/lltm.cpp"])]
    if torch.cuda.is_available() and (CUDA_HOME is not None):
        ext_modules.append(
            CUDAExtension(
                "monai._C_CUDA",
                ["monai/networks/extensions/lltm/lltm_cuda.cpp", "monai/networks/extensions/lltm/lltm_cuda_kernel.cu"],
            )
        )
    return ext_modules


def get_cmds():
    cmds = versioneer.get_cmdclass()
    try:
        from torch.utils.cpp_extension import BuildExtension

        cmds.update({"build_ext": BuildExtension})
    except ImportError:
        warnings.warn("torch cpp_extension module not found.")
    return cmds


setup(
    version=versioneer.get_version(),
    cmdclass=get_cmds(),
    packages=find_packages(exclude=("docs", "examples", "tests", "research")),
    zip_safe=False,
    package_data={"monai": ["py.typed"]},
    ext_modules=get_extensions(),
)
