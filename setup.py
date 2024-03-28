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

import os
import warnings

from setuptools import find_packages, setup

import versioneer

# Determine whether to build C++ and CUDA extensions
BUILD_MONAI = os.getenv("BUILD_MONAI", "0") == "1"
FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"

# Define extension modules
ext_modules = []
if BUILD_MONAI:
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension

        print(f"Setup with torch {torch.__version__}")

        ext_modules.append(
            CUDAExtension(
                name="monai._C",
                sources=["monai/csrc/*.cpp", "monai/csrc/*.cu"],
                include_dirs=["monai/csrc"],
                define_macros=[("WITH_CUDA", None)],
            )
        )

    except ImportError as e:
        warnings.warn(f"Failed to import Torch: {e}")

# Configure setup options
setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=("docs", "examples", "tests")),
    zip_safe=False,
    package_data={"monai": ["py.typed", "monai/csrc/*"]},
    ext_modules=ext_modules,
)
