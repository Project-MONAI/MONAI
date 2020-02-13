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

import numpy as np
import torch

import monai

try:
    import ignite
    ignite_version = ignite.__version__
except ImportError:
    ignite_version = 'NOT INSTALLED'

export = monai.utils.export("monai.config")


@export
def get_config_values():
    output = OrderedDict()

    output["MONAI version"] = monai.__version__
    output["Python version"] = sys.version.replace("\n", " ")
    output["Numpy version"] = np.version.full_version
    output["Pytorch version"] = torch.__version__
    output["Ignite version"] = ignite_version

    return output


@export
def print_config(file=sys.stdout):
    for kv in get_config_values().items():
        print("%s: %s" % kv, file=file, flush=True)


@export
def set_visible_devices(*dev_inds):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, dev_inds))
