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

export = monai.utils.export("monai.application.config")


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
