# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from glob import glob
from os import makedirs, path
from shutil import rmtree

from torch.utils.cpp_extension import load

dir_path = path.dirname(path.realpath(__file__))


def load_module(module_name, defines=None, verbose_build=False, force_build=False):

    if defines is None:
        define_args = []
    else:
        define_args = [f"-D {key}={defines[key]}" for key in defines]

    module_dir = path.join(dir_path, module_name)

    assert path.exists(module_dir), f"No extention module named {module_name}"

    build_tag = "_".join(str(v) for v in defines.values())
    build_name = "build" if build_tag == "" else f"build_{build_tag}"
    build_dir = path.join(module_dir, "build", build_name)

    if force_build and path.exists(build_dir) or path.exists(path.join(build_dir, "lock")):
        rmtree(build_dir)

    if not path.exists(build_dir):
        makedirs(build_dir)

    source = glob(path.join(module_dir, "**/*.cpp"), recursive=True)
    source += glob(path.join(module_dir, "**/*.cu"), recursive=True)

    module = load(
        name=module_name,
        sources=source,
        extra_cflags=define_args,
        extra_cuda_cflags=define_args,
        build_directory=build_dir,
        verbose=verbose_build,
    )

    return module
