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

import os
import platform
import re
import sys
from collections import OrderedDict

import numpy as np
import torch

import monai
from monai.utils import OptionalImportError, get_package_version, optional_import

try:
    import itk  # type: ignore

    itk_version = itk.Version.GetITKVersion()
    del itk
except (ImportError, AttributeError):
    itk_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    _, HAS_EXT = optional_import("monai._C")
    USE_COMPILED = HAS_EXT and os.getenv("BUILD_MONAI", "0") == "1"
except (OptionalImportError, ImportError, AttributeError):
    HAS_EXT = USE_COMPILED = False

psutil, has_psutil = optional_import("psutil")
psutil_version = psutil.__version__ if has_psutil else "NOT INSTALLED or UNKNOWN VERSION."

__all__ = [
    "print_config",
    "get_system_info",
    "print_system_info",
    "get_gpu_info",
    "print_gpu_info",
    "print_debug_info",
    "USE_COMPILED",
]


def get_config_values():
    """
    Read the package versions into a dictionary.
    """
    output = OrderedDict()

    output["MONAI"] = monai.__version__
    output["Numpy"] = np.version.full_version
    output["Pytorch"] = torch.__version__

    return output


def get_optional_config_values():
    """
    Read the optional package versions into a dictionary.
    """
    output = OrderedDict()

    output["Pytorch Ignite"] = get_package_version("ignite")
    output["Nibabel"] = get_package_version("nibabel")
    output["scikit-image"] = get_package_version("skimage")
    output["Pillow"] = get_package_version("PIL")
    output["Tensorboard"] = get_package_version("tensorboard")
    output["gdown"] = get_package_version("gdown")
    output["TorchVision"] = get_package_version("torchvision")
    output["ITK"] = itk_version
    output["tqdm"] = get_package_version("tqdm")
    output["lmdb"] = get_package_version("lmdb")
    output["psutil"] = psutil_version

    return output


def print_config(file=sys.stdout):
    """
    Print the package versions to `file`.

    Args:
        file: `print()` text stream file. Defaults to `sys.stdout`.
    """
    for k, v in get_config_values().items():
        print(f"{k} version: {v}", file=file, flush=True)
    print(f"MONAI flags: HAS_EXT = {HAS_EXT}, USE_COMPILED = {USE_COMPILED}")
    print(f"MONAI rev id: {monai.__revision_id__}")

    print("\nOptional dependencies:", file=file, flush=True)
    for k, v in get_optional_config_values().items():
        print(f"{k} version: {v}", file=file, flush=True)
    print("\nFor details about installing the optional dependencies, please visit:", file=file, flush=True)
    print(
        "    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies\n",
        file=file,
        flush=True,
    )


def set_visible_devices(*dev_inds):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, dev_inds))


def _dict_append(in_dict, key, fn):
    try:
        in_dict[key] = fn() if callable(fn) else fn
    except BaseException:
        in_dict[key] = "UNKNOWN for given OS"


def get_system_info() -> OrderedDict:
    """
    Get system info as an ordered dictionary.
    """
    output: OrderedDict = OrderedDict()

    _dict_append(output, "System", platform.system)
    if output["System"] == "Windows":
        _dict_append(output, "Win32 version", platform.win32_ver)
        if hasattr(platform, "win32_edition"):
            _dict_append(output, "Win32 edition", platform.win32_edition)  # type:ignore[attr-defined]
    elif output["System"] == "Darwin":
        _dict_append(output, "Mac version", lambda: platform.mac_ver()[0])
    else:
        with open("/etc/os-release", "r") as rel_f:
            linux_ver = re.search(r'PRETTY_NAME="(.*)"', rel_f.read())
        if linux_ver:
            _dict_append(output, "Linux version", lambda: linux_ver.group(1))

    _dict_append(output, "Platform", platform.platform)
    _dict_append(output, "Processor", platform.processor)
    _dict_append(output, "Machine", platform.machine)
    _dict_append(output, "Python version", platform.python_version)

    if not has_psutil:
        _dict_append(output, "`psutil` missing", lambda: "run `pip install monai[psutil]`")
    else:
        p = psutil.Process()
        with p.oneshot():
            _dict_append(output, "Process name", p.name)
            _dict_append(output, "Command", p.cmdline)
            _dict_append(output, "Open files", p.open_files)
            _dict_append(output, "Num physical CPUs", lambda: psutil.cpu_count(logical=False))
            _dict_append(output, "Num logical CPUs", lambda: psutil.cpu_count(logical=True))
            _dict_append(output, "Num usable CPUs", lambda: len(psutil.Process().cpu_affinity()))
            _dict_append(output, "CPU usage (%)", lambda: psutil.cpu_percent(percpu=True))
            _dict_append(output, "CPU freq. (MHz)", lambda: round(psutil.cpu_freq(percpu=False)[0]))
            _dict_append(
                output,
                "Load avg. in last 1, 5, 15 mins (%)",
                lambda: [round(x / psutil.cpu_count() * 100, 1) for x in psutil.getloadavg()],
            )
            _dict_append(output, "Disk usage (%)", lambda: psutil.disk_usage(os.getcwd()).percent)
            _dict_append(
                output,
                "Avg. sensor temp. (Celsius)",
                lambda: np.round(
                    np.mean([item.current for sublist in psutil.sensors_temperatures().values() for item in sublist], 1)
                ),
            )
            mem = psutil.virtual_memory()
            _dict_append(output, "Total physical memory (GB)", lambda: round(mem.total / 1024 ** 3, 1))
            _dict_append(output, "Available memory (GB)", lambda: round(mem.available / 1024 ** 3, 1))
            _dict_append(output, "Used memory (GB)", lambda: round(mem.used / 1024 ** 3, 1))

    return output


def print_system_info(file=sys.stdout) -> None:
    """
    Print system info to `file`. Requires the optional library, `psutil`.

    Args:
        file: `print()` text stream file. Defaults to `sys.stdout`.
    """
    if not has_psutil:
        print("`psutil` required for `print_system_info`", file=file, flush=True)
    else:
        for k, v in get_system_info().items():
            print(f"{k}: {v}", file=file, flush=True)


def get_gpu_info() -> OrderedDict:

    output: OrderedDict = OrderedDict()

    num_gpus = torch.cuda.device_count()
    _dict_append(output, "Num GPUs", lambda: num_gpus)

    _dict_append(output, "Has CUDA", lambda: bool(torch.cuda.is_available()))

    if output["Has CUDA"]:
        _dict_append(output, "CUDA version", lambda: torch.version.cuda)
    cudnn_ver = torch.backends.cudnn.version()
    _dict_append(output, "cuDNN enabled", lambda: bool(cudnn_ver))

    if cudnn_ver:
        _dict_append(output, "cuDNN version", lambda: cudnn_ver)

    if num_gpus > 0:
        _dict_append(output, "Current device", torch.cuda.current_device)
        if hasattr(torch.cuda, "get_arch_list"):  # get_arch_list is new in torch 1.7.1
            _dict_append(output, "Library compiled for CUDA architectures", torch.cuda.get_arch_list)

    for gpu in range(num_gpus):
        gpu_info = torch.cuda.get_device_properties(gpu)
        _dict_append(output, f"GPU {gpu} Name", lambda: gpu_info.name)
        _dict_append(output, f"GPU {gpu} Is integrated", lambda: bool(gpu_info.is_integrated))
        _dict_append(output, f"GPU {gpu} Is multi GPU board", lambda: bool(gpu_info.is_multi_gpu_board))
        _dict_append(output, f"GPU {gpu} Multi processor count", lambda: gpu_info.multi_processor_count)
        _dict_append(output, f"GPU {gpu} Total memory (GB)", lambda: round(gpu_info.total_memory / 1024 ** 3, 1))
        _dict_append(
            output, f"GPU {gpu} Cached memory (GB)", lambda: round(torch.cuda.memory_reserved(gpu) / 1024 ** 3, 1)
        )
        _dict_append(
            output, f"GPU {gpu} Allocated memory (GB)", lambda: round(torch.cuda.memory_allocated(gpu) / 1024 ** 3, 1)
        )
        _dict_append(output, f"GPU {gpu} CUDA capability (maj.min)", lambda: f"{gpu_info.major}.{gpu_info.minor}")

    return output


def print_gpu_info(file=sys.stdout) -> None:
    """
    Print GPU info to `file`.

    Args:
        file: `print()` text stream file. Defaults to `sys.stdout`.
    """
    for k, v in get_gpu_info().items():
        print(f"{k}: {v}", file=file, flush=True)


def print_debug_info(file=sys.stdout) -> None:
    """
    Print config (installed dependencies, etc.) and system info for debugging.

    Args:
        file: `print()` text stream file. Defaults to `sys.stdout`.
    """
    print("================================", file=file, flush=True)
    print("Printing MONAI config...", file=file, flush=True)
    print("================================", file=file, flush=True)
    print_config(file)
    print("\n================================", file=file, flush=True)
    print("Printing system config...")
    print("================================", file=file, flush=True)
    print_system_info(file)
    print("\n================================", file=file, flush=True)
    print("Printing GPU config...")
    print("================================", file=file, flush=True)
    print_gpu_info(file)


if __name__ == "__main__":
    print_debug_info()
