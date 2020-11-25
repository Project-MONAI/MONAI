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
import platform
import sys
from collections import OrderedDict

import numpy as np
import torch

import monai
from monai.utils import OptionalImportError, optional_import

try:
    import ignite

    ignite_version = ignite.__version__
    del ignite
except (ImportError, AttributeError):
    ignite_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import nibabel

    nibabel_version = nibabel.__version__
    del nibabel
except (ImportError, AttributeError):
    nibabel_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import skimage

    skimage_version = skimage.__version__
    del skimage
except (ImportError, AttributeError):
    skimage_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import PIL

    PIL_version = PIL.__version__
    del PIL
except (ImportError, AttributeError):
    PIL_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import tensorboard

    tensorboard_version = tensorboard.__version__
    del tensorboard
except (ImportError, AttributeError):
    tensorboard_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import gdown

    gdown_version = gdown.__version__
    del gdown
except (ImportError, AttributeError):
    gdown_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import torchvision

    torchvision_version = torchvision.__version__
    del torchvision
except (ImportError, AttributeError):
    torchvision_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import itk  # type: ignore

    itk_version = itk.Version.GetITKVersion()
    del itk
except (ImportError, AttributeError):
    itk_version = "NOT INSTALLED or UNKNOWN VERSION."

try:
    import tqdm

    tqdm_version = tqdm.__version__
    del tqdm
except (ImportError, AttributeError):
    tqdm_version = "NOT INSTALLED or UNKNOWN VERSION."


try:
    import lmdb  # type: ignore

    lmdb_version = lmdb.__version__
    del lmdb
except (ImportError, AttributeError):
    lmdb_version = "NOT INSTALLED or UNKNOWN VERSION."


try:
    _, HAS_EXT = optional_import("monai._C")
    USE_COMPILED = HAS_EXT and os.getenv("BUILD_MONAI", "0") == "1"
except (OptionalImportError, ImportError, AttributeError):
    HAS_EXT = USE_COMPILED = False

psutil, has_psutil = optional_import("psutil")


def get_config_values():
    """
    Read the package versions into a dictionary.
    """
    output = OrderedDict()

    output["MONAI"] = monai.__version__
    output["Python"] = sys.version.replace("\n", " ")
    output["OS"] = f"{platform.system()} ({platform.release()})"
    output["Numpy"] = np.version.full_version
    output["Pytorch"] = torch.__version__

    return output


def get_optional_config_values():
    """
    Read the optional package versions into a dictionary.
    """
    output = OrderedDict()

    output["Pytorch Ignite"] = ignite_version
    output["Nibabel"] = nibabel_version
    output["scikit-image"] = skimage_version
    output["Pillow"] = PIL_version
    output["Tensorboard"] = tensorboard_version
    output["gdown"] = gdown_version
    output["TorchVision"] = torchvision_version
    output["ITK"] = itk_version
    output["tqdm"] = tqdm_version
    output["lmdb"] = lmdb_version

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


def get_torch_version_tuple():
    """
    Returns:
        tuple of ints represents the pytorch major/minor version.
    """
    return tuple((int(x) for x in torch.__version__.split(".")[:2]))


def _dict_append(in_dict, key, fn):
    try:
        in_dict[key] = fn()
    except BaseException:
        in_dict[key] = "UNKNOWN for given OS"


def get_system_info(file=sys.stdout) -> OrderedDict:
    """
    Get system info as an ordered dictionary.
    """
    output: OrderedDict = OrderedDict()

    if not has_psutil:
        print("`psutil` required for `get_system_info", file=file, flush=True)
        return output

    p = psutil.Process()
    with p.oneshot():
        _dict_append(output, "Process name", lambda: p.name())
        _dict_append(output, "Command", lambda: p.cmdline())
        _dict_append(output, "Open files", lambda: p.open_files())
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
            "Avg. sensor temp. (°C)",
            lambda: np.mean([item.current for sublist in psutil.sensors_temperatures().values() for item in sublist]),
        )

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
        for k, v in get_system_info(file).items():
            print(f"{k}: {v}", file=file, flush=True)


def get_gpu_info() -> OrderedDict:

    output: OrderedDict = OrderedDict()

    num_gpus = torch.cuda.device_count()
    _dict_append(output, "Num GPUs", lambda: num_gpus)
    if num_gpus > 0:
        _dict_append(output, "Current device", lambda: torch.cuda.current_device())
        _dict_append(output, "Library compiled for CUDA architectures", lambda: torch.cuda.get_arch_list())
    for gpu in range(num_gpus):
        _dict_append(output, "Info for GPU", lambda: gpu)
        gpu_info = torch.cuda.get_device_properties(gpu)
        _dict_append(output, "\tName", lambda: gpu_info.name)
        _dict_append(output, "\tIs integrated", lambda: bool(gpu_info.is_integrated))
        _dict_append(output, "\tIs multi GPU board", lambda: bool(gpu_info.is_multi_gpu_board))
        _dict_append(output, "\tMulti processor count", lambda: gpu_info.multi_processor_count)
        _dict_append(output, "\tTotal memory (GB)", lambda: round(gpu_info.total_memory / 1024 ** 3, 1))
        _dict_append(output, "\tCached memory (GB)", lambda: round(torch.cuda.memory_reserved(gpu) / 1024 ** 3, 1))
        _dict_append(output, "\tAllocated memory (GB)", lambda: round(torch.cuda.memory_allocated(gpu) / 1024 ** 3, 1))
        _dict_append(output, "\tCUDA capability (maj.min)", lambda: f"{gpu_info.major}.{gpu_info.minor}")

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
