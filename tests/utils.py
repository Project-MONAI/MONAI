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
import tempfile
import unittest
from io import BytesIO
from subprocess import PIPE, Popen
from urllib.error import ContentTooShortError, HTTPError, URLError

import numpy as np
import torch

from monai.data import create_test_image_2d, create_test_image_3d
from monai.utils import optional_import, set_determinism

nib, _ = optional_import("nibabel")

quick_test_var = "QUICKTEST"


def test_pretrained_networks(network, input_param, device):
    try:
        net = network(**input_param).to(device)
    except (URLError, HTTPError, ContentTooShortError) as e:
        raise unittest.SkipTest(e)
    return net


def skip_if_quick(obj):
    """
    Skip the unit tests if environment variable `quick_test_var=true`.
    For example, the user can skip the relevant tests by setting ``export QUICKTEST=true``.
    """
    is_quick = os.environ.get(quick_test_var, "").lower() == "true"

    return unittest.skipIf(is_quick, "Skipping slow tests")(obj)


class SkipIfNoModule(object):
    """Decorator to be used if test should be skipped
    when optional module is not present."""

    def __init__(self, module_name):
        self.module_name = module_name
        self.module_missing = not optional_import(self.module_name)[1]

    def __call__(self, obj):
        return unittest.skipIf(self.module_missing, f"optional module not present: {self.module_name}")(obj)


def skip_if_no_cuda(obj):
    """
    Skip the unit tests if torch.cuda.is_available is False
    """
    return unittest.skipIf(not torch.cuda.is_available(), "Skipping CUDA-based tests")(obj)


def skip_if_windows(obj):
    """
    Skip the unit tests if platform is win32
    """
    return unittest.skipIf(sys.platform == "win32", "Skipping tests on Windows")(obj)


def make_nifti_image(array, affine=None):
    """
    Create a temporary nifti image on the disk and return the image name.
    User is responsible for deleting the temporary file when done with it.
    """
    if affine is None:
        affine = np.eye(4)
    test_image = nib.Nifti1Image(array, affine)

    temp_f, image_name = tempfile.mkstemp(suffix=".nii.gz")
    nib.save(test_image, image_name)
    os.close(temp_f)
    return image_name


class NumpyImageTestCase2D(unittest.TestCase):
    im_shape = (128, 64)
    input_channels = 1
    output_channels = 4
    num_classes = 3

    def setUp(self):
        im, msk = create_test_image_2d(self.im_shape[0], self.im_shape[1], 4, 20, 0, self.num_classes)

        self.imt = im[None, None]
        self.seg1 = (msk[None, None] > 0).astype(np.float32)
        self.segn = msk[None, None]


class TorchImageTestCase2D(NumpyImageTestCase2D):
    def setUp(self):
        NumpyImageTestCase2D.setUp(self)
        self.imt = torch.tensor(self.imt)
        self.seg1 = torch.tensor(self.seg1)
        self.segn = torch.tensor(self.segn)


class NumpyImageTestCase3D(unittest.TestCase):
    im_shape = (64, 48, 80)
    input_channels = 1
    output_channels = 4
    num_classes = 3

    def setUp(self):
        im, msk = create_test_image_3d(self.im_shape[0], self.im_shape[1], self.im_shape[2], 4, 20, 0, self.num_classes)

        self.imt = im[None, None]
        self.seg1 = (msk[None, None] > 0).astype(np.float32)
        self.segn = msk[None, None]


class TorchImageTestCase3D(NumpyImageTestCase3D):
    def setUp(self):
        NumpyImageTestCase3D.setUp(self)
        self.imt = torch.tensor(self.imt)
        self.seg1 = torch.tensor(self.seg1)
        self.segn = torch.tensor(self.segn)


def test_script_save(net, *inputs, eval_nets=True, device=None):
    """
    Test the ability to save `net` as a Torchscript object, reload it, and apply inference. The value `inputs` is
    forward-passed through the original and loaded copy of the network and their results returned. Both `net` and its
    reloaded copy are set to evaluation mode if `eval_nets` is True. The forward pass for both is done without
    gradient accumulation.

    The test will be performed with CUDA if available, else CPU.
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to device
    inputs = [i.to(device) for i in inputs]

    scripted = torch.jit.script(net)
    buffer = scripted.save_to_buffer()
    reloaded_net = torch.jit.load(BytesIO(buffer)).to(device)
    net.to(device)

    if eval_nets:
        net.eval()
        reloaded_net.eval()

    with torch.no_grad():
        set_determinism(seed=0)
        result1 = net(*inputs)
        result2 = reloaded_net(*inputs)
        set_determinism(seed=None)
    # When using e.g., VAR, we will produce a tuple of outputs.
    # Hence, convert all to tuples and then compare all elements.
    if not isinstance(result1, tuple):
        result1 = (result1,)
        result2 = (result2,)

    for i, (r1, r2) in enumerate(zip(result1, result2)):
        if None not in (r1, r2):  # might be None
            np.testing.assert_allclose(
                r1.detach().cpu().numpy(),
                r2.detach().cpu().numpy(),
                rtol=1e-5,
                atol=0,
                err_msg=f"failed on comparison number: {i}",
            )


def query_memory(n=2):
    """
    Find best n idle devices and return a string of device ids.
    """
    bash_string = "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader,nounits"

    try:
        p1 = Popen(bash_string.split(), stdout=PIPE)
        output, error = p1.communicate()
        free_memory = [x.split(",") for x in output.decode("utf-8").split("\n")[:-1]]
        free_memory = np.asarray(free_memory, dtype=np.float).T
        ids = np.lexsort(free_memory)[:n]
    except (FileNotFoundError, TypeError, IndexError):
        ids = range(n) if isinstance(n, int) else []
    return ",".join([f"{int(x)}" for x in ids])


if __name__ == "__main__":
    print(query_memory())
