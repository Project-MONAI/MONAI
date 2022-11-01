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
from monai.utils import convert_to_tensor, TransformBackends

from monai.transforms.lazy.functional import apply
from monai.transforms.meta_matrix import MetaMatrix, MatrixFactory


def single_2d_transform_cases():
    f = MatrixFactory(2, TransformBackends.TORCH, "cpu")

    cases = [
        (
            torch.randn((1, 32, 32)),
            [MetaMatrix(f.rotate_euler(torch.pi / 4).matrix, {"id": "rotate"})],
            (1, 32, 32)
        ),
        (
            torch.randn((1, 16, 16)),
            [MetaMatrix(f.rotate_euler(torch.pi / 4).matrix,
                        {"id": "rotate", "shape_override": (1, 45, 45)})],
            (1, 45, 45)
        )
    ]

    return cases


class TestApply(unittest.TestCase):

    def _test_apply_impl(self, tensor, pending_transforms):
        print(tensor.shape)
        result = apply(tensor, pending_transforms)
        self.assertListEqual(result[1], pending_transforms)

    def _test_apply_metatensor_impl(self, tensor, pending_transforms, expected_shape, pending_as_parameter):
        tensor_ = convert_to_tensor(tensor, track_meta=True)
        if pending_as_parameter:
            result, transforms = apply(tensor_, pending_transforms)
        else:
            for p in pending_transforms:
                tensor_.push_pending_operation(p)
            result, transforms = apply(tensor_)

        self.assertEqual(result.shape, expected_shape)


    SINGLE_TRANSFORM_CASES = single_2d_transform_cases()

    def test_apply_single_transform(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_impl(*case)

    def test_apply_single_transform_metatensor(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_metatensor_impl(*case, False)

    def test_apply_single_transform_metatensor_override(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_metatensor_impl(*case, True)


if __name__ == '__main__':
    unittest.main()
