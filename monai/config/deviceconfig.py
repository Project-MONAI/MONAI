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
import sys
from collections import OrderedDict

import monai
import numpy as np
import torch


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


def get_config_values():
    """
    Read the package versions into a dictionary.
    """
    output = OrderedDict()

    output["MONAI"] = monai.__version__
    output["Python"] = sys.version.replace("\n", " ")
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

    return output


def print_config(file=sys.stdout):
    """
    Print the package versions to `file`.
    Defaults to `sys.stdout`.
    """
    for k, v in get_config_values().items():
        print(f"{k} version: {v}", file=file, flush=True)

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
    return tuple([int(x) for x in torch.__version__.split(".")[:2]])
