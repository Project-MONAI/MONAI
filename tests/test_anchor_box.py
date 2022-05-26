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

from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from monai.utils import optional_import
from tests.utils import assert_allclose

torch_anchor_utils, _ = optional_import("torchvision.models.detection.anchor_utils", version="0.12")
image_list, _ = optional_import("torchvision.models.detection.image_list", version="0.12")


TEST_CASES_2D = []
TEST_CASES_2D.append(
    [
        {"sizes": ((10, 12, 14, 16), (20, 24, 28, 32)), "aspect_ratios": ((1.0, 0.5, 2.0), (1.0, 0.5, 2.0))},
        (5, 3, 128, 128),
        ((5, 7, 64, 32), (5, 7, 32, 16)),
    ]
)


class TestAnchorGenerator(unittest.TestCase):
    @parameterized.expand(TEST_CASES_2D)
    def test_anchor_2d(self, input_param, image_shape, feature_maps_shapes):
        # test it behaves the same with torchvision for 2d
        anchor = AnchorGenerator(**input_param, indexing="xy")
        anchor_ref = torch_anchor_utils.AnchorGenerator(**input_param)
        for a, a_f in zip(anchor.cell_anchors, anchor_ref.cell_anchors):
            assert_allclose(a, a_f, type_test=True, device_test=False, atol=1e-3)
        for a, a_f in zip(anchor.num_anchors_per_location(), anchor_ref.num_anchors_per_location()):
            assert_allclose(a, a_f, type_test=True, device_test=False, atol=1e-3)

        grid_sizes = [[2, 2], [1, 1]]
        strides = [[torch.tensor(1), torch.tensor(2)], [torch.tensor(2), torch.tensor(4)]]
        for a, a_f in zip(anchor.grid_anchors(grid_sizes, strides), anchor_ref.grid_anchors(grid_sizes, strides)):
            assert_allclose(a, a_f, type_test=True, device_test=False, atol=1e-3)

        images = torch.rand(image_shape)
        feature_maps = tuple(torch.rand(fs) for fs in feature_maps_shapes)
        result = anchor(images, feature_maps)
        result_ref = anchor_ref(image_list.ImageList(images, ([123, 122],)), feature_maps)
        for a, a_f in zip(result, result_ref):
            assert_allclose(a, a_f, type_test=True, device_test=False, atol=0.1)


if __name__ == "__main__":
    unittest.main()
