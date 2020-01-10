import os, sys
from collections import OrderedDict
import monai
import numpy as np
import torch

try:
    import ignite
    ignite_version=ignite.__version__
except ImportError:
    ignite_version='NOT INSTALLED'

export = monai.utils.export("monai.application.config")


@export
def getConfigValues():
    output = OrderedDict()

    output["MONAI version"] = monai.__version__
    output["Python version"] = sys.version.replace("\n", " ")
    output["Numpy version"] = np.version.full_version
    output["Pytorch version"] = torch.__version__
    output["Ignite version"] = ignite_version

    return output


@export
def printConfig(file=sys.stdout):
    for kv in getConfigValues().items():
        print("%s: %s" % kv, file=file, flush=True)


@export
def setVisibleDevices(*devInds):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devInds))
