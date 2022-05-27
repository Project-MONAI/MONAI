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
from parameterized import parameterized

from monai.apps.detection.utils.box_coder import BoxCoder
from monai.transforms import CastToType
from tests.utils import assert_allclose

TESTS = []

TESTS.append([torch.tensor([[0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]])])


class TestBoxTransform(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, boxes):
        box_coder = BoxCoder(weights=[1, 1, 1, 1, 1, 1])
        test_dtype = [torch.float32, torch.float16]
        for dtype in test_dtype:
            gt_boxes = CastToType(dtype=dtype)(boxes)
            proposals = gt_boxes + torch.rand(gt_boxes.shape)
            rel_gt_boxes = box_coder.encode_single(gt_boxes, proposals)
            gt_back = box_coder.decode_single(rel_gt_boxes, proposals)
            assert_allclose(gt_back, gt_boxes, type_test=True, device_test=True, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
