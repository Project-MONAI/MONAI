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
import unittest
import torch
import numpy as np

from monai.utils.generateddata import create_test_image_2d

quick_test_var = "QUICKTEST"


def skip_if_quick(obj):
    is_quick = os.environ.get(quick_test_var, "").lower() == "true"

    return unittest.skipIf(is_quick, "Skipping slow tests")(obj)


class ImageTestCase(unittest.TestCase):
    im_shape = (128, 128)
    input_channels = 1
    output_channels = 4
    num_classes = 3

    def setUp(self):
        im, msk = create_test_image_2d(self.im_shape[0], self.im_shape[1], 4, 20, 0, self.num_classes)

        self.imt = torch.tensor(im[None, None])

        self.seg1 = torch.tensor((msk[None, None] > 0).astype(np.float32))
        self.segn = torch.tensor(msk[None, None])
