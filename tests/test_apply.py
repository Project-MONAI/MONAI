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
from monai.utils import convert_to_tensor

from monai.transforms.lazy.functional import apply
from monai.transforms.meta_matrix import MetaMatrix



def rotate_45_2d():
    t = torch.eye(3)
    t[:, 0] = torch.FloatTensor([0, -1, 0])
    t[:, 1] = torch.FloatTensor([1, 0, 0])
    return t


class TestApply(unittest.TestCase):

    def _test_apply_impl(self, tensor, pending_transforms):
        print(tensor.shape)
        result = apply(tensor, pending_transforms)
        self.assertListEqual(result[1], pending_transforms)

    def _test_apply_metatensor_impl(self, tensor, pending_transforms, pending_as_parameter):
        tensor_ = convert_to_tensor(tensor)
        if pending_as_parameter:
            result = apply(tensor_, pending_transforms)
        else:
            for p in pending_transforms:
                # TODO: cannot do the next part until the MetaTensor PR #5107 is in
                # tensor_.push_pending(p)
                raise NotImplementedError()

    SINGLE_TRANSFORM_CASES = [
        (torch.randn((1, 16, 16)), [MetaMatrix(rotate_45_2d(), {"id": "rotate"})])
    ]

    def test_apply_single_transform(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_impl(*case)

    def test_apply_single_transform_metatensor(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_metatensor_impl(*case, False)

    def test_apply_single_transform_metatensor_override(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_metatensor_impl(*case, True)
