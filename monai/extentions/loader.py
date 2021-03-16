from os import path, makedirs
from shutil import rmtree
from glob import glob
from torch.utils.cpp_extension import load

dir_path = path.dirname(path.realpath(__file__))

def load_module(module_name, defines={}, verbose_build=False, force_build=False):

    define_args = [f'-D {key}={defines[key]}' for key in defines]

    module_dir = path.join(dir_path, module_name)

    assert path.exists(module_dir), f"No extention module named {module_name}"

    build_tag = '_'.join(str(v) for v in defines.values())
    build_name = 'build' if build_tag == '' else f'build_{build_tag}'
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
        verbose=verbose_build
    )

    return module
    