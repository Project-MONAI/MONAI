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

import unittest


import torch

from monai.transforms.utility.functional import resample
from monai.utils import convert_to_tensor


from tests.utils import get_arange_img


def rotate_45_2d():
    t = torch.eye(3)
    t[:, 0] = torch.FloatTensor([0, -1, 0])
    t[:, 1] = torch.FloatTensor([1, 0, 0])
    return t


class TestResampleFunction(unittest.TestCase):

    def _test_resample_function_impl(self, img, matrix):
        result = resample(convert_to_tensor(img), matrix)
        print(result)

    RESAMPLE_FUNCTION_CASES = [
        (get_arange_img((1, 16, 16)), rotate_45_2d())
    ]

    def test_resample_function(self):
        for case in self.RESAMPLE_FUNCTION_CASES:
            self._test_resample_function_impl(*case)
