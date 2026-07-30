#! /usr/bin/env python

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

"""
Script for checking various elements of the runtime environment and printing a large amount of diagnostic information.

This is meant to be used for debugging environments used with MONAI, but doesn't directly need MONAI itself. It will
print information about the environment, including trying to get installed packages, test PyTorch with CUDA, and then
have MONAI print its debugging information if no errors encountered. If MONAI is not installed this script should still
work and produce useful information. Only standard libraries are needed in case a bare environment is being used.

This can be run as a program with the following options to see all outputs:

    python check_env.py --env --envvars --monai

With no options at all this can be used to test that PyTorch is installed and can move a tensor to a device. This is
useful when creating a fresh test environment and MONAI isn't present yet but it's good practice to valid PyTorch.

This can also be used remotely with only Python installed to get current environment diagnostic info:

    curl https://raw.githubusercontent.com/Project-MONAI/MONAI/refs/heads/dev/monai/config/check_env.py | python
"""

from io import StringIO
import os
import sys
import platform
import multiprocessing
import subprocess
import shutil
import argparse
import getpass
from functools import partial

DESC = """
Script for checking various elements of the runtime environment and printing a large amount of diagnostic information.
This is used for debugging your environment by printing out various system statistics and diagnostic information. It
checks PyTorch and MONAI are installed and functioning. A typical use case is with the `--env` and `--monai` options.
"""


USER = getpass.getuser()
HOST = platform.node()
efprint = partial(print, flush=True, file=sys.stderr)


def fprint(*args, **kwargs):
    """
    Print with flushing, replacing the username and hostname values with placeholders for better anonymization.
    """
    kwargs["flush"] = True
    content = " ".join(map(str, args))
    content = content.replace(USER, "<user>").replace(HOST, "<host>")
    print(content, **kwargs)


def print_platform():
    """
    Print basic platform information.
    """
    fprint(platform.platform())
    fprint("uname:", list(platform.uname()))
    fprint("CPU:", platform.processor(), "Count:", multiprocessing.cpu_count())
    fprint("Python:", sys.executable, platform.python_implementation(), platform.python_version())


def print_environment_vars():
    """
    Print all environment variables other than a few known pointless ones.
    """
    fprint("Environment:")
    for k, v in os.environ.items():
        if k not in ("LS_COLORS", "PS1", "PS2"):
            fprint(f"  {k}:", v)


def print_environment():
    """
    Print the installed environment using `conda` or `pip`, fail if neither are present.
    """
    try:
        cmd = ("conda", "env", "export") if shutil.which("conda") else ("pip", "list")

        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        fprint(result.decode())
        return True
    except Exception as e:
        efprint(f"Exception encountered getting environment with conda/pip: {e}")
        return False


def check_torch():
    """
    Check PyTorch is installed and a tensor can be created.
    """
    try:
        import torch

        t = torch.rand(2, 3) * 5
        fprint("PyTorch:", torch.__version__, torch.__path__)
        fprint("Test tensor:", t.flatten())
        return True
    except ImportError:
        efprint("PyTorch not installed")
        return False


def check_torch_cuda():
    """
    Check CUDA capability in PyTorch by moving a tensor to each available device.
    """
    import torch

    dcount = torch.cuda.device_count()
    fprint("CUDA version:", torch.version.cuda)
    fprint("PyTorch GPU Count:", dcount)

    try:
        for d in range(dcount):
            fprint(f"  {torch.cuda.get_device_properties(d)}")
            t = torch.rand(2, 3).to(torch.device(f"cuda:{d}")) * 5
            fprint("Test tensor:", t.flatten())
        return True
    except Exception as e:
        efprint(f"PyTorch encountered exception creating GPU tensor on device {d}: {e}")
        return False


def check_monai():
    """
    Check MONAI by importing it then printing its debug info.
    """
    try:
        import monai

        out = StringIO()
        monai.config.deviceconfig.print_debug_info(file=out)
        out.seek(0)
        fprint(out.read())
        return True
    except ImportError:
        efprint("MONAI not installed")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="check_env.py", description=DESC.strip())
    parser.add_argument("--env", default=False, action="store_true", help="Print environment info")
    parser.add_argument("--envvars", default=False, action="store_true", help="Include environment variables")
    parser.add_argument("--monai", default=False, action="store_true", help="Print MONAI info")
    args = parser.parse_args()

    fprint("=" * 10, "Platform Info", "=" * 10)
    print_platform()

    if args.env:
        fprint("=" * 10, "Checking Environment", "=" * 10)
        if args.envvars:
            print_environment_vars()
        print_environment()

    fprint("=" * 10, "Checking PyTorch", "=" * 10)

    if not check_torch():
        efprint("Exiting early, no valid PyTorch install found.")
        sys.exit(1)

    if not check_torch_cuda():
        efprint("Exiting early, PyTorch encountered CUDA error.")
        sys.exit(1)

    if args.monai:
        fprint("=" * 10, "Checking MONAI", "=" * 10)
        check_monai()
