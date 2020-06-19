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
import tempfile
import unittest
from subprocess import PIPE, Popen

import numpy as np
import torch
from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from monai.utils import optional_import

nib, _ = optional_import("nibabel")

quick_test_var = "QUICKTEST"


def skip_if_quick(obj):
    is_quick = os.environ.get(quick_test_var, "").lower() == "true"

    return unittest.skipIf(is_quick, "Skipping slow tests")(obj)


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


def expect_failure_if_no_gpu(test):
    if not torch.cuda.is_available():
        return unittest.expectedFailure(test)
    else:
        return test


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
