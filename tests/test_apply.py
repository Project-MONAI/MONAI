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

import numpy as np

import torch

from monai.transforms.lazy.functional import apply
from monai.transforms.meta_matrix import MetaMatrix


def get_img(size, dtype=torch.float32, offset=0):
    img = torch.zeros(size, dtype=dtype)
    if len(size) == 2:
        for j in range(size[0]):
            for i in range(size[1]):
                img[j, i] = i + j * size[0] + offset
    else:
        for k in range(size[0]):
            for j in range(size[1]):
                for i in range(size[2]):
                    img[k, j, i] = i + j * size[0] + k * size[0] * size[1]
    return np.expand_dims(img, 0)


def rotate_45_2D():
    t = torch.eye(3)
    t[:, 0] = torch.FloatTensor([0, -1, 0])
    t[:, 1] = torch.FloatTensor([1, 0, 0])
    return t


class TestApply(unittest.TestCase):

    def _test_apply_impl(self, tensor, pending_transforms):
        print(tensor.shape)
        # for m in pending_transforms:
        #     print(m.matrix)
        #     print(m.metadata)
        result = apply(tensor, pending_transforms)
        print(result)

    SINGLE_TRANSFORM_CASES = [
        (torch.randn((1, 16, 16)), [MetaMatrix(rotate_45_2D(), {"id", "rotate"})])
    ]

    def test_apply_single_transform(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_impl(*case)

