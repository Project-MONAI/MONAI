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

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lltm_cuda",
    ext_modules=[
        CUDAExtension("lltm_cuda", [
            "monai/networks/extensions/lltm/lltm_cuda.cpp",
            "monai/networks/extensions/lltm/lltm_cuda_kernel.cu",
        ]),
    ],
    cmdclass={
        "build_ext": BuildExtension
    })
